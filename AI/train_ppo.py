"""
train_ppo.py — Architecture améliorée :
  • Action space : 90 actions discrètes (lookup table pruné)
  • Rewards     : distance exponentielle, touch hauteur, flip reset, boost, team spirit
  • Obs         : ExtendedObs 101 floats
  • Réseau      : DiscreteFF 101→512→512→90 (via rlgym-ppo)
  • Adversaire  : self-play (les deux agents s'entraînent ensemble)
"""

import math
import os
import random
from pathlib import Path
from typing import Any, Dict

import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from rlgym.api import ActionParser, AgentID, RewardFunction
from rlgym.api import RLGym
from rlgym.rocket_league import common_values
from rlgym.rocket_league.action_parsers import RepeatAction
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.common_values import (
    ORANGE_TEAM, BLUE_TEAM, BACK_NET_Y,
    CAR_MAX_SPEED, CAR_MAX_ANG_VEL,
    BALL_RADIUS, CEILING_Z, GOAL_HEIGHT,
)
from rlgym.rocket_league.done_conditions import (
    AnyCondition, GoalCondition, NoTouchTimeoutCondition, TimeoutCondition,
)
from rlgym.rocket_league.obs_builders import DefaultObs
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.state_mutators import (
    FixedTeamSizeMutator, KickoffMutator, MutatorSequence,
)

from rlgym_ppo import Learner
from rlgym_ppo.util import RLGymV2GymWrapper
from rlgym_ppo.util import reporting as _reporting

os.chdir(Path(__file__).resolve().parent)

# ---------------------------------------------------------------------------
# Hyperparamètres
# ---------------------------------------------------------------------------
TOTAL_TIMESTEPS = 500_000_000   # fresh start
N_PROC          = 20
ACTION_REPEAT   = 8

# Cosine LR schedule
LR_INITIAL = 5e-4
LR_FINAL   = 5e-5
LR_CYCLE   = 100_000_000

# Goals
BLUE_GOAL   = np.array([0.0, -BACK_NET_Y, 0.0], dtype=np.float32)
ORANGE_GOAL = np.array([0.0,  BACK_NET_Y, 0.0], dtype=np.float32)

# ---------------------------------------------------------------------------
# TensorBoard + LR scheduler
# ---------------------------------------------------------------------------
tb_writer = SummaryWriter(log_dir="data/tensorboard/run8", flush_secs=10)
_learner  = None

_original_report = _reporting.report_metrics

def _tb_report_metrics(loggable_metrics, debug_metrics, wandb_run=None):
    _original_report(loggable_metrics, debug_metrics, wandb_run)
    step = loggable_metrics.get("Cumulative Timesteps", 0)

    if _learner is not None:
        cycle_pos = step % LR_CYCLE
        t_cycle   = cycle_pos / LR_CYCLE
        new_lr    = LR_FINAL + 0.5 * (LR_INITIAL - LR_FINAL) * (1.0 + math.cos(math.pi * t_cycle))
        for pg in _learner.ppo_learner.policy_optimizer.param_groups:
            pg['lr'] = new_lr
        for pg in _learner.ppo_learner.value_optimizer.param_groups:
            pg['lr'] = new_lr
        tb_writer.add_scalar("Learning Rate", new_lr, global_step=step)

    for key, val in loggable_metrics.items():
        if isinstance(val, (int, float)):
            tb_writer.add_scalar(key, val, global_step=step)
    tb_writer.flush()

_reporting.report_metrics = _tb_report_metrics


# ---------------------------------------------------------------------------
# ExtendedObs — 101 floats (inchangé)
# ---------------------------------------------------------------------------
class ExtendedObs(DefaultObs):
    def get_obs_space(self, agent: AgentID):
        shape_type, size = super().get_obs_space(agent)
        return shape_type, size + 9

    def _build_obs(self, agent: AgentID, state: GameState, shared_info: dict) -> np.ndarray:
        base_obs = super()._build_obs(agent, state, shared_info)
        car     = state.cars[agent]
        invert  = (car.team_num == ORANGE_TEAM)
        ball    = state.inverted_ball  if invert else state.ball
        physics = car.inverted_physics if invert else car.physics

        own_goal   = np.array([0.0,  BACK_NET_Y, 0.0], dtype=np.float32)
        their_goal = np.array([0.0, -BACK_NET_Y, 0.0], dtype=np.float32)

        def unit(v):
            n = np.linalg.norm(v)
            return v / n if n > 1e-6 else v

        extra = np.concatenate([
            unit(their_goal - ball.position),
            unit(own_goal   - ball.position),
            unit(ball.position - physics.position),
        ])
        return np.concatenate([base_obs, extra])


# ---------------------------------------------------------------------------
# DiscreteActionParser — 90 actions discrètes
# ---------------------------------------------------------------------------
class DiscreteActionParser(ActionParser):
    def __init__(self):
        self._lookup_table = self._make_lookup_table()

    @staticmethod
    def _make_lookup_table() -> np.ndarray:
        actions = []
        # Ground : throttle × steer × boost × handbrake (sans boost sans plein throttle)
        for throttle in (-1, 0, 1):
            for steer in (-1, 0, 1):
                for boost in (0, 1):
                    for handbrake in (0, 1):
                        if boost == 1 and throttle != 1:
                            continue
                        actions.append([throttle or boost, steer, 0, steer, 0, 0, boost, handbrake])
        # Aerial : pitch × yaw × roll × jump × boost (sans yaw+jump, sans noop)
        for pitch in (-1, 0, 1):
            for yaw in (-1, 0, 1):
                for roll in (-1, 0, 1):
                    for jump in (0, 1):
                        for boost in (0, 1):
                            if jump == 1 and yaw != 0:
                                continue
                            if pitch == roll == jump == 0:
                                continue
                            handbrake = int(jump == 1 and (pitch != 0 or yaw != 0 or roll != 0))
                            actions.append([boost, yaw, pitch, yaw, roll, jump, boost, handbrake])
        return np.array(actions, dtype=np.float32)

    def get_action_space(self, agent: AgentID):
        return 'discrete', len(self._lookup_table)

    def reset(self, agents, initial_state, shared_info) -> None:
        pass

    def parse_actions(self, actions: Dict[AgentID, np.ndarray],
                      state: GameState,
                      shared_info: Dict[str, Any]) -> Dict[AgentID, np.ndarray]:
        parsed = {}
        for agent, action in actions.items():
            parsed[agent] = self._lookup_table[int(action)]
        return parsed


# Wrapper gym pour espace d'action discret
class DiscreteGymWrapper(RLGymV2GymWrapper):
    def __init__(self, rlgym_env):
        super().__init__(rlgym_env)
        act_space = list(rlgym_env.action_spaces.values())[0][1]
        if isinstance(act_space, int):
            self.action_space = gym.spaces.Discrete(n=act_space)


# ---------------------------------------------------------------------------
# AdvancedReward
# ---------------------------------------------------------------------------
def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-6 or nb < 1e-6:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _height_activation(z: float) -> float:
    return float(np.cbrt((z - 150.0) / CEILING_Z))


def _dist_to_closest_wall(x: float, y: float) -> float:
    dist_side = abs(4096 - abs(x))
    dist_back = abs(5120 - abs(y))
    # Distance au coin coupé
    x1, y1, x2, y2 = 4096 - 1152, 5120, 4096, 5120 - 1152
    A, B = abs(x) - x1, abs(y) - y1
    C, D = x2 - x1, y2 - y1
    len_sq = C * C + D * D
    param = (A * C + B * D) / len_sq if len_sq != 0 else -1
    xx = x1 + np.clip(param, 0, 1) * C
    yy = y1 + np.clip(param, 0, 1) * D
    dist_corner = math.sqrt((abs(x) - xx) ** 2 + (abs(y) - yy) ** 2)
    return min(dist_side, dist_back, dist_corner)


class AdvancedReward(RewardFunction[AgentID, GameState, float]):
    """
    Conservé  : distance exponentielle, touch hauteur (racine cubique),
                flip reset, boost gain/lose, ang_vel, demo, team spirit.
    """

    def __init__(
        self,
        team_spirit: float = 0.6,
        goal_w:         float = 10.0,
        goal_dist_w:    float = 10.0,
        demo_w:         float = 8.0,
        dist_w:         float = 0.25,
        align_w:        float = 0.25,
        boost_gain_w:   float = 1.5,
        boost_lose_w:   float = 0.8,
        ang_vel_w:      float = 0.005,
        touch_grass_w:  float = 0.005,
        touch_height_w: float = 3.0,
        touch_accel_w:  float = 0.5,
        flip_reset_w:   float = 10.0,
        opponent_punish_w: float = 1.0,
    ):
        self._ts            = team_spirit
        self._goal_w        = goal_w
        self._goal_dist_w   = goal_dist_w
        self._demo_w        = demo_w
        self._dist_w        = dist_w
        self._align_w       = align_w
        self._boost_gain_w  = boost_gain_w
        self._boost_lose_w  = boost_lose_w
        self._ang_vel_w     = ang_vel_w
        self._touch_grass_w = touch_grass_w
        self._touch_height_w = touch_height_w
        self._touch_accel_w = touch_accel_w
        self._flip_reset_w  = flip_reset_w
        self._opp_w         = opponent_punish_w

        # État précédent
        self._prev_ball_vel      : np.ndarray = None
        self._prev_boosts        : dict = {}
        self._prev_has_flip      : dict = {}
        self._prev_demo          : dict = {}
        self._prev_state_quality : float = 0.0
        self._prev_player_qual   : dict = {}

    # ------------------------------------------------------------------
    def _state_quality(self, state: GameState) -> float:
        ball = state.ball.position
        return float(0.5 * self._goal_dist_w * (
            np.exp(-np.linalg.norm(ORANGE_GOAL - ball) / CAR_MAX_SPEED)
            - np.exp(-np.linalg.norm(BLUE_GOAL  - ball) / CAR_MAX_SPEED)
        ))

    def _player_quality(self, agent_id: AgentID, state: GameState) -> float:
        car  = state.cars[agent_id]
        pos  = car.physics.position
        ball = state.ball.position
        liu  = float(np.exp(-np.linalg.norm(ball - pos) / 1410.0))
        goal = ORANGE_GOAL if car.team_num == BLUE_TEAM else BLUE_GOAL
        align = _cosine_sim(ball - pos, goal - pos)
        if car.team_num == ORANGE_TEAM:
            align *= -1
        return self._dist_w * liu + self._align_w * align

    # ------------------------------------------------------------------
    def reset(self, agents, initial_state: GameState, shared_info=None) -> None:
        self._prev_ball_vel = np.array(initial_state.ball.linear_velocity, dtype=np.float32)
        self._prev_state_quality = self._state_quality(initial_state)
        for a in agents:
            car = initial_state.cars[a]
            self._prev_boosts[a]   = float(car.boost_amount)
            self._prev_has_flip[a] = bool(car.has_flip)
            self._prev_demo[a]     = bool(car.is_demoed)
            self._prev_player_qual[a] = self._player_quality(a, initial_state)

    # ------------------------------------------------------------------
    def get_rewards(self, agents, state: GameState,
                    is_terminated, _is_truncated, _shared_info):
        sq  = self._state_quality(state)
        pq  = {a: self._player_quality(a, state) for a in agents}
        curr_ball_vel = np.array(state.ball.linear_velocity, dtype=np.float32)
        delta_ball_vel = float(np.linalg.norm(curr_ball_vel - self._prev_ball_vel))

        rewards = {}
        for agent_id in agents:
            car = state.cars[agent_id]
            pos = car.physics.position
            r   = 0.0

            # ---- Qualité d'état (balle qui se rapproche du bon but) ----
            delta_sq = sq - self._prev_state_quality
            r += delta_sq if car.team_num == BLUE_TEAM else -delta_sq

            # ---- Qualité joueur (positionnement + alignement) ----
            r += pq[agent_id] - self._prev_player_qual.get(agent_id, 0.0)

            # ---- Touch rewards ----
            if car.ball_touches > 0:
                ball_h = state.ball.position[2]
                avg_h  = 0.5 * (pos[2] + ball_h)
                h0 = _height_activation(0.0)
                h1 = _height_activation(float(CEILING_Z))
                hx = _height_activation(avg_h)
                height_factor = ((hx - h0) / (h1 - h0 + 1e-6)) ** 2
                wall_dist_f = 1 - np.exp(
                    -_dist_to_closest_wall(float(pos[0]), float(pos[1])) / CAR_MAX_SPEED
                )
                r += self._touch_height_w * height_factor * (1 + wall_dist_f)

                # Flip reset (air dribble setup)
                has_flip_now = bool(car.has_flip)
                had_flip     = self._prev_has_flip.get(agent_id, False)
                if (has_flip_now and not had_flip
                        and pos[2] > 3 * BALL_RADIUS
                        and np.linalg.norm(state.ball.position - pos) < 2 * BALL_RADIUS):
                    up = car.physics.rotation_mtx[2]
                    if _cosine_sim(state.ball.position - pos, -up) > 0.9:
                        r += self._flip_reset_w

                # Accélération balle (tir fort = plus de reward)
                r += self._touch_accel_w * (1 - height_factor) * delta_ball_vel / CAR_MAX_SPEED

            # ---- Boost management ----
            prev_b    = self._prev_boosts.get(agent_id, float(car.boost_amount))
            boost_now = float(car.boost_amount)
            boost_diff = (math.sqrt(max(boost_now, 0.0)) - math.sqrt(max(prev_b, 0.0)))
            if boost_diff >= 0:
                r += self._boost_gain_w * boost_diff
            elif pos[2] < GOAL_HEIGHT:
                r += self._boost_lose_w * boost_diff * (1 - pos[2] / GOAL_HEIGHT)

            # ---- Angular velocity (encourage l'exploration aérienne) ----
            ang_vel_norm = float(np.linalg.norm(car.physics.angular_velocity)) / CAR_MAX_ANG_VEL
            r += ang_vel_norm * self._ang_vel_w

            # ---- Touch grass penalty ----
            if car.on_ground and pos[2] < BALL_RADIUS:
                r -= self._touch_grass_w

            # ---- Demo : pénalité sur la victime ----
            is_demo  = bool(car.is_demoed)
            was_demo = self._prev_demo.get(agent_id, False)
            if is_demo and not was_demo:
                r -= self._demo_w / 2

            rewards[agent_id] = r

        # ---- Demo : bonus pour le demoeur ----
        for agent_id in agents:
            for other_id in agents:
                if other_id == agent_id:
                    continue
                other = state.cars[other_id]
                if bool(other.is_demoed) and not self._prev_demo.get(other_id, False):
                    rewards[agent_id] += self._demo_w / 2

        # ---- Goal reward ----
        any_terminated = any(is_terminated.values()) if isinstance(is_terminated, dict) else bool(is_terminated)
        if any_terminated:
            ball_y = float(state.ball.position[1])
            if ball_y > BACK_NET_Y * 0.5:        # balle dans le but orange
                for a in agents:
                    rewards[a] += self._goal_w if state.cars[a].team_num == BLUE_TEAM else -self._goal_w
            elif ball_y < -BACK_NET_Y * 0.5:     # balle dans le but blue
                for a in agents:
                    rewards[a] += self._goal_w if state.cars[a].team_num == ORANGE_TEAM else -self._goal_w

        # ---- Team spirit blending (0.6) ----
        blue_ids   = [a for a in agents if state.cars[a].team_num == BLUE_TEAM]
        orange_ids = [a for a in agents if state.cars[a].team_num == ORANGE_TEAM]
        if blue_ids and orange_ids:
            br = np.array([rewards[a] for a in blue_ids])
            or_ = np.array([rewards[a] for a in orange_ids])
            bm  = float(np.nan_to_num(br.mean()))
            om  = float(np.nan_to_num(or_.mean()))
            for i, a in enumerate(blue_ids):
                rewards[a] = (1 - self._ts) * br[i] + self._ts * bm - self._opp_w * om
            for i, a in enumerate(orange_ids):
                rewards[a] = (1 - self._ts) * or_[i] + self._ts * om - self._opp_w * bm

        # ---- Mise à jour de l'état précédent ----
        self._prev_ball_vel      = curr_ball_vel.copy()
        self._prev_state_quality = sq
        self._prev_player_qual   = pq
        for a in agents:
            car = state.cars[a]
            self._prev_boosts[a]   = float(car.boost_amount)
            self._prev_has_flip[a] = bool(car.has_flip)
            self._prev_demo[a]     = bool(car.is_demoed)

        return rewards


# ---------------------------------------------------------------------------
# Population Self-Play Wrapper
# ---------------------------------------------------------------------------
_N_ACTIONS = len(DiscreteActionParser._make_lookup_table())   # 90


class PopulationSelfPlayWrapper(gym.Env):
    """
    Self-play avec pool de checkpoints passés.
    Blue  = policy courante entraînée par PPO.
    Orange = checkpoint gelé, pioché aléatoirement dans l'historique.
             Swappé toutes les `swap_every` steps.
             Avant qu'un checkpoint existe → actions aléatoires.
    """

    def __init__(self, inner_env: DiscreteGymWrapper,
                 checkpoint_dir: str,
                 swap_every: int = 3_000_000,
                 pool_size: int = 10,
                 device: str = "cuda"):
        self._inner          = inner_env
        self._ckpt_dir       = Path(checkpoint_dir)
        self._swap_every     = swap_every
        self._pool_size      = pool_size
        self._device         = torch.device(device)
        self._steps          = 0
        self._opponent       = None          # DiscreteFF gelée, ou None
        self._blue_idx       = 0
        self._orange_idx     = 1

        n_obs = inner_env.observation_space.shape[0]
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(n_obs,), dtype=np.float32
        )
        self.action_space = inner_env.action_space

    # ------------------------------------------------------------------
    def _find_indices(self):
        state = self._inner.rlgym_env.state
        for idx, agent_id in self._inner.agent_map.items():
            if state.cars[agent_id].is_blue:
                self._blue_idx   = idx
            else:
                self._orange_idx = idx

    def _load_random_opponent(self):
        """Charge un checkpoint aléatoire parmi les `pool_size` derniers."""
        step_dirs = sorted(
            (p for p in self._ckpt_dir.iterdir()
             if p.is_dir() and p.name.isdigit()),
            key=lambda p: int(p.name),
        )
        if not step_dirs:
            self._opponent = None
            return

        pool   = step_dirs[-self._pool_size:]
        chosen = random.choice(pool) / "PPO_POLICY.pt"
        if not chosen.exists():
            self._opponent = None
            return

        from rlgym_ppo.ppo import DiscreteFF
        state_dict = torch.load(chosen, map_location="cpu")
        obs_size   = state_dict[next(iter(state_dict))].shape[1]
        model      = DiscreteFF(obs_size, 90, [1024, 1024], "cpu")
        model.load_state_dict(state_dict)
        model.eval()
        self._opponent = model
        print(f"[SelfPlay] Nouvel adversaire chargé : {chosen.parent.name} steps")

    # ------------------------------------------------------------------
    def reset(self):
        obs_all = self._inner.reset()
        self._find_indices()
        return obs_all[self._blue_idx]

    def step(self, blue_action):
        self._steps += 1
        if self._steps % self._swap_every == 0:
            self._load_random_opponent()

        # Action orange
        if self._opponent is None:
            opp_action = random.randint(0, _N_ACTIONS - 1)
        else:
            orange_obs = self._inner.obs_buffer[self._orange_idx]
            obs_t = torch.FloatTensor(orange_obs).unsqueeze(0)
            with torch.no_grad():
                act, _ = self._opponent.get_action(obs_t, deterministic=False)
            opp_action = int(np.asarray(act).flat[0])

        b_idx   = int(np.asarray(blue_action).flat[0])
        actions = np.array([0, 0], dtype=np.int32)
        actions[self._blue_idx]   = b_idx
        actions[self._orange_idx] = opp_action

        obs_all, rews, done, truncated, info = self._inner.step(actions)
        return obs_all[self._blue_idx], rews[self._blue_idx], done, truncated, info

    def close(self):
        self._inner.close()


# ---------------------------------------------------------------------------
# Environment Builder
# ---------------------------------------------------------------------------
def build_rlgym_v2_env():
    action_parser = RepeatAction(DiscreteActionParser(), repeats=ACTION_REPEAT)

    termination_condition = GoalCondition()
    truncation_condition  = AnyCondition(
        NoTouchTimeoutCondition(timeout_seconds=30),
        TimeoutCondition(timeout_seconds=300),
    )

    obs_builder = ExtendedObs(
        zero_padding=1,
        pos_coef=np.asarray([
            1 / common_values.SIDE_WALL_X,
            1 / BACK_NET_Y,
            1 / CEILING_Z,
        ]),
        ang_coef     = 1 / np.pi,
        lin_vel_coef = 1 / CAR_MAX_SPEED,
        ang_vel_coef = 1 / CAR_MAX_ANG_VEL,
        boost_coef   = 1 / 100.0,
    )

    state_mutator = MutatorSequence(
        FixedTeamSizeMutator(blue_size=1, orange_size=1),
        KickoffMutator(),
    )

    rlgym_env = RLGym(
        state_mutator     = state_mutator,
        obs_builder       = obs_builder,
        action_parser     = action_parser,
        reward_fn         = AdvancedReward(),
        termination_cond  = termination_condition,
        truncation_cond   = truncation_condition,
        transition_engine = RocketSimEngine(),
    )

    return PopulationSelfPlayWrapper(
        DiscreteGymWrapper(rlgym_env),
        checkpoint_dir = "data/checkpoints/prl-run-v2",
        swap_every     = 3_000_000,
        pool_size      = 10,
        device         = "cuda",
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def main():
    global _learner

    min_inference_size = max(1, int(round(N_PROC * 0.9)))

    _learner = Learner(
        build_rlgym_v2_env,
        n_proc                  = N_PROC,
        min_inference_size      = min_inference_size,
        metrics_logger          = None,
        device                  = "cuda",
        checkpoints_save_folder = "data/checkpoints/prl-run-v2",
        add_unix_timestamp      = False,
        n_checkpoints_to_keep   = 10,
        ppo_batch_size          = 50_000,
        policy_layer_sizes      = [1024, 1024],
        critic_layer_sizes      = [1024, 1024],
        ts_per_iteration        = 100_000,
        exp_buffer_size         = 300_000,
        ppo_minibatch_size      = 10_000,
        ppo_ent_coef            = 0.01,
        policy_lr               = LR_INITIAL,
        critic_lr               = LR_INITIAL,
        ppo_epochs              = 3,
        standardize_returns     = True,
        standardize_obs         = False,
        save_every_ts           = 500_000,
        timestep_limit          = TOTAL_TIMESTEPS,
        log_to_wandb            = False,
    )

    _learner.learn()
    tb_writer.close()


if __name__ == "__main__":
    main()
