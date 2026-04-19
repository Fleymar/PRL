import math
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from rlgym.api import ActionParser, AgentID, RewardFunction
from rlgym.api import RLGym
from rlgym.rocket_league import common_values
from rlgym.rocket_league.action_parsers import RepeatAction
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.common_values import ORANGE_TEAM, BACK_NET_Y, BLUE_TEAM
from rlgym.rocket_league.done_conditions import (
    AnyCondition,
    GoalCondition,
    NoTouchTimeoutCondition,
    TimeoutCondition,
)
from rlgym.rocket_league.obs_builders import DefaultObs
from rlgym.rocket_league.reward_functions import GoalReward
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.state_mutators import (
    FixedTeamSizeMutator,
    KickoffMutator,
    MutatorSequence,
)

from rlgym_ppo import Learner
from rlgym_ppo.util import RLGymV2GymWrapper
from rlgym_ppo.util import reporting as _reporting

import gym

# Répertoire de travail = AI/
os.chdir(Path(__file__).resolve().parent)

# ---------------------------------------------------------------------------
# Hyperparamètres globaux
# ---------------------------------------------------------------------------
TOTAL_TIMESTEPS = 2_000_000_000
N_PROC          = 14
ACTION_REPEAT   = 8   # doit correspondre à RepeatAction(repeats=8)

# Offset curriculum — permet de reprendre directement en Phase 3
# après un restart depuis un checkpoint avancé
CURRICULUM_OFFSET = 100_000_000

# Phases du curriculum (en timesteps globaux estimés)
PHASE1_END = 20_000_000   # 0  → 20M : apprendre à bouger et toucher la balle
PHASE2_END = 60_000_000   # 20M→ 60M : orienter la balle vers le but
                           # 60M→500M : marquer / défendre

# Learning rate schedule
# Cosine annealing avec période de 100M steps → remonte le LR périodiquement
# pour sortir des local optima, au lieu d'une décroissance linéaire monotone.
LR_INITIAL  = 5e-4
LR_FINAL    = 5e-5   # floor du cosine (était 1e-5, trop bas)
LR_CYCLE    = 100_000_000  # redémarre tous les 100M steps

# ---------------------------------------------------------------------------
# TensorBoard + LR Scheduler
# ---------------------------------------------------------------------------
tb_writer = SummaryWriter(log_dir="data/tensorboard/run7", flush_secs=10)
_learner  = None

_original_report = _reporting.report_metrics

def _tb_report_metrics(loggable_metrics, debug_metrics, wandb_run=None):
    _original_report(loggable_metrics, debug_metrics, wandb_run)

    step = loggable_metrics.get("Cumulative Timesteps", 0)

    if _learner is not None:
        # Cosine annealing avec restarts tous les LR_CYCLE steps
        # Permet de sortir des plateaux sans LR monotone décroissant
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
# Custom Obs Builder (DefaultObs + 9 vecteurs relatifs)
# ---------------------------------------------------------------------------
class ExtendedObs(DefaultObs):
    def get_obs_space(self, agent: AgentID):
        shape_type, size = super().get_obs_space(agent)
        return shape_type, size + 9

    def _build_obs(self, agent: AgentID, state: GameState, shared_info: dict) -> np.ndarray:
        base_obs = super()._build_obs(agent, state, shared_info)

        car     = state.cars[agent]
        invert  = (car.team_num == ORANGE_TEAM)
        ball    = state.inverted_ball    if invert else state.ball
        physics = car.inverted_physics   if invert else car.physics

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
# MultiDiscrete Action Parser
# ---------------------------------------------------------------------------
_BINS = [
    [-1.0,  0.0,  1.0],   # throttle
    [-1.0,  0.0,  1.0],   # steer
    [-1.0,  0.0,  1.0],   # pitch
    [-1.0,  0.0,  1.0],   # yaw
    [-1.0,  0.0,  1.0],   # roll
    [ 0.0,  1.0],          # jump
    [ 0.0,  1.0],          # boost
    [ 0.0,  1.0],          # handbrake
]
_BIN_SIZES = [len(b) for b in _BINS]


class MultiDiscreteActionParser(ActionParser):
    def get_action_space(self, agent: AgentID):
        return 'multi_discrete', _BIN_SIZES

    def reset(self, agents, initial_state, shared_info) -> None:
        pass

    def parse_actions(self, actions: Dict[AgentID, np.ndarray],
                      state: GameState,
                      shared_info: Dict[str, Any]) -> Dict[AgentID, np.ndarray]:
        parsed = {}
        for agent, action in actions.items():
            parsed[agent] = np.array(
                [_BINS[i][int(action[i])] for i in range(len(_BINS))],
                dtype=np.float32,
            )
        return parsed


class MultiDiscreteGymWrapper(RLGymV2GymWrapper):
    def __init__(self, rlgym_env):
        super().__init__(rlgym_env)
        act_space = list(rlgym_env.action_spaces.values())[0][1]
        if isinstance(act_space, (list, tuple)):
            self.action_space = gym.spaces.MultiDiscrete(act_space)


# ---------------------------------------------------------------------------
# Reward Functions
# ---------------------------------------------------------------------------

class SpeedTowardBallReward(RewardFunction[AgentID, GameState, float]):
    """
    Vitesse du bot vers la balle, clampée à [0, 1].
    Jamais négative → un flip latéral/arrière donne 0, impossible à farmer.
    """
    def reset(self, agents, initial_state, shared_info) -> None:
        pass

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        rewards = {}
        ball_pos = state.ball.position
        for agent in agents:
            car  = state.cars[agent]
            diff = ball_pos - car.physics.position
            dist = np.linalg.norm(diff)
            if dist < 1e-6:
                rewards[agent] = 0.0
                continue
            vel_toward = float(np.dot(car.physics.linear_velocity, diff / dist))
            rewards[agent] = max(0.0, vel_toward / common_values.CAR_MAX_SPEED)
        return rewards


class VelocityBallToGoalReward(RewardFunction[AgentID, GameState, float]):
    def reset(self, agents, initial_state, shared_info) -> None:
        pass

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        rewards = {}
        ball_pos = state.ball.position
        ball_vel = state.ball.linear_velocity
        for agent in agents:
            car      = state.cars[agent]
            goal_pos = np.array(
                [0.0, common_values.BACK_NET_Y if car.team_num == BLUE_TEAM
                 else -common_values.BACK_NET_Y, 0.0], dtype=np.float32
            )
            diff = goal_pos - ball_pos
            dist = np.linalg.norm(diff)
            if dist < 1e-6:
                rewards[agent] = 0.0
                continue
            rewards[agent] = float(np.dot(ball_vel, diff / dist)) / common_values.BALL_MAX_SPEED
        return rewards


class TouchDeltaVReward(RewardFunction[AgentID, GameState, float]):
    """
    Reward proportionnel au changement de vitesse de la balle lors d'un touch.
    Un vrai tir puissant → reward élevé. Un push léger → presque 0.
    Le delta est calculé une seule fois par step (partagé entre agents)
    mais appliqué uniquement aux agents qui ont touché la balle.
    """
    def __init__(self):
        self._prev_ball_vel = None
        self._cached_delta  = 0.0  # delta calculé une seule fois par step

    def reset(self, agents, initial_state, shared_info) -> None:
        self._prev_ball_vel = None
        self._cached_delta  = 0.0

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        curr_vel = np.array(state.ball.linear_velocity, dtype=np.float32)

        if self._prev_ball_vel is None:
            self._prev_ball_vel = curr_vel.copy()
            return {agent: 0.0 for agent in agents}

        # Calculer le delta une seule fois pour ce step
        delta_v = float(np.linalg.norm(curr_vel - self._prev_ball_vel))
        self._prev_ball_vel = curr_vel.copy()

        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            if car.ball_touches > 0 and delta_v > 0.0:
                rewards[agent] = delta_v / common_values.BALL_MAX_SPEED
            else:
                rewards[agent] = 0.0
        return rewards


class HighVelocitySaveReward(RewardFunction[AgentID, GameState, float]):
    SPEED_THRESHOLD = 800.0

    def reset(self, agents, initial_state, shared_info) -> None:
        pass

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        rewards = {}
        ball_vel = state.ball.linear_velocity
        for agent in agents:
            car = state.cars[agent]
            if car.team_num == BLUE_TEAM:
                vel_toward_our_goal = -ball_vel[1]
            else:
                vel_toward_our_goal =  ball_vel[1]
            if vel_toward_our_goal > self.SPEED_THRESHOLD and car.ball_touches > 0:
                rewards[agent] = vel_toward_our_goal / common_values.BALL_MAX_SPEED
            else:
                rewards[agent] = 0.0
        return rewards


class DefensivePenaltyReward(RewardFunction[AgentID, GameState, float]):
    def reset(self, agents, initial_state, shared_info) -> None:
        pass

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        rewards = {}
        ball_y = state.ball.position[1]
        for agent in agents:
            car = state.cars[agent]
            if car.team_num == BLUE_TEAM:
                in_our_half = ball_y < 0.0
            else:
                in_our_half = ball_y > 0.0
            rewards[agent] = -1.0 if in_our_half else 0.0
        return rewards


# ---------------------------------------------------------------------------
# Curriculum Combined Reward
# ---------------------------------------------------------------------------

def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * max(0.0, min(1.0, t))


class CurriculumCombinedReward(RewardFunction[AgentID, GameState, float]):
    """
    Phase 1 (0→20M)   : poids fixes — apprendre à bouger et toucher
    Phase 2 (20→60M)  : speed_toward diminue progressivement, ball_to_goal monte
    Phase 3 (60→300M) : focus marquer/défendre, dense reward maintenu pour le signal
    """

    def __init__(self, n_proc: int = N_PROC):
        self._n_proc      = n_proc
        self._local_steps = 0
        self._speed       = SpeedTowardBallReward()
        self._btg         = VelocityBallToGoalReward()
        self._touch       = TouchDeltaVReward()
        self._goal        = GoalReward()
        self._save        = HighVelocitySaveReward()
        self._defensive   = DefensivePenaltyReward()

    @property
    def _global_steps(self) -> int:
        # _local_steps est incrémenté par step d'env (un seul env par instance).
        # On multiplie par ACTION_REPEAT (8) pour avoir des env-steps réels,
        # puis on approxime les agent-steps globaux.
        # N_PROC workers × steps_par_env ≈ total agent timesteps.
        return self._local_steps * ACTION_REPEAT + CURRICULUM_OFFSET

    def _get_weights(self) -> dict:
        gs = self._global_steps

        if gs <= PHASE1_END:
            # Poids fixes — cible stable pour l'apprentissage initial
            return dict(
                speed     = 2.0,
                btg       = 0.5,
                touch     = 3.0,
                goal      = 15.0,
                save      = 0.0,
                defensive = 0.0,
            )
        elif gs <= PHASE2_END:
            t = (gs - PHASE1_END) / (PHASE2_END - PHASE1_END)
            return dict(
                speed     = _lerp(2.0,  0.0, t),
                btg       = _lerp(0.5,  3.0, t),
                touch     = _lerp(3.0,  1.0, t),
                goal      = _lerp(15.0, 20.0, t),
                save      = _lerp(0.0,  2.0, t),
                defensive = _lerp(0.0,  0.002, t),
            )
        else:
            t = min((gs - PHASE2_END) / (TOTAL_TIMESTEPS - PHASE2_END), 1.0)
            return dict(
                # Floors élevés : le critic doit toujours avoir un signal dense
                # pour estimer la valeur des états. Sans ça → gradient = bruit → collapse.
                speed     = _lerp(2.0,  0.5, t),   # jamais < 0.5
                btg       = _lerp(3.0,  2.0, t),   # jamais < 2.0
                touch     = _lerp(1.5,  0.8, t),   # jamais < 0.8
                goal      = _lerp(20.0, 25.0, t),  # 20-25 max, pas 40-50
                save      = _lerp(2.0,  3.0, t),
                defensive = _lerp(0.002, 0.003, t),
            )

    def reset(self, agents, initial_state, shared_info) -> None:
        self._speed.reset(agents, initial_state, shared_info)
        self._btg.reset(agents, initial_state, shared_info)
        self._touch.reset(agents, initial_state, shared_info)
        self._goal.reset(agents, initial_state, shared_info)
        self._save.reset(agents, initial_state, shared_info)
        self._defensive.reset(agents, initial_state, shared_info)

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        self._local_steps += 1
        w = self._get_weights()

        r_speed = self._speed.get_rewards(agents, state, is_terminated, is_truncated, shared_info)
        r_btg   = self._btg.get_rewards(agents, state, is_terminated, is_truncated, shared_info)
        r_touch = self._touch.get_rewards(agents, state, is_terminated, is_truncated, shared_info)
        r_goal  = self._goal.get_rewards(agents, state, is_terminated, is_truncated, shared_info)
        r_save  = self._save.get_rewards(agents, state, is_terminated, is_truncated, shared_info)
        r_def   = self._defensive.get_rewards(agents, state, is_terminated, is_truncated, shared_info)

        return {
            agent: (
                w['speed']     * r_speed[agent] +
                w['btg']       * r_btg[agent]   +
                w['touch']     * r_touch[agent] +
                w['goal']      * r_goal[agent]  +
                w['save']      * r_save[agent]  +
                w['defensive'] * r_def[agent]
            )
            for agent in agents
        }


# ---------------------------------------------------------------------------
# Heuristic Opponent (remplace le self-play)
# ---------------------------------------------------------------------------

def _heuristic_action(state, agent_id) -> np.ndarray:
    """
    Adversaire heuristique agressif : fonce sur la balle avec boost,
    essaie de tirer vers le but adverse.
    Retourne indices MultiDiscrete [throttle, steer, pitch, yaw, roll, jump, boost, handbrake].
    """
    car  = state.cars[agent_id]
    # Coordonnées dans le repère "toujours blue" pour cohérence
    phys = car.inverted_physics if car.is_orange else car.physics
    ball = state.inverted_ball   if car.is_orange else state.ball

    car_pos  = phys.position        # (3,)
    ball_pos = ball.position        # (3,)

    # Vecteurs d'orientation (RocketSim : row 0 = forward, row 1 = right)
    right = phys.rotation_mtx[1]

    # But adverse (dans le repère inverted : toujours à Y négatif)
    their_goal = np.array([0.0, -BACK_NET_Y, 0.0], dtype=np.float32)

    to_ball      = ball_pos - car_pos
    dist_to_ball = float(np.linalg.norm(to_ball))

    # Direction cible : approche directe si loin, position derrière balle si proche
    if dist_to_ball > 500 or dist_to_ball < 1e-6:
        target_dir = to_ball / (dist_to_ball + 1e-6)
    else:
        # Se positionner entre le but et la balle pour shooter
        ball_to_goal = their_goal - ball_pos
        n = float(np.linalg.norm(ball_to_goal))
        if n > 1e-6:
            # Point 300 uu derrière la balle (côté de notre voiture)
            ideal_pos = ball_pos - ball_to_goal / n * 300
            d = ideal_pos - car_pos
            dist_d = float(np.linalg.norm(d))
            target_dir = d / (dist_d + 1e-6)
        else:
            target_dir = to_ball / (dist_to_ball + 1e-6)

    # Steer : dot produit avec vecteur droit
    dot_right = float(np.dot(right, target_dir))
    if   dot_right >  0.15: steer_idx = 2   # virage droite
    elif dot_right < -0.15: steer_idx = 0   # virage gauche
    else:                   steer_idx = 1   # tout droit

    # Pitch : neutre (garder simple, éviter le saut intempestif)
    pitch_idx = 1

    # Toujours plein gaz + boost agressif
    return np.array([2, steer_idx, pitch_idx, steer_idx, 1, 0, 1, 0], dtype=np.int32)
    # [throttle=+1, steer, pitch=0, yaw=steer, roll=0, jump=0, boost=1, handbrake=0]


class HeuristicOpponentWrapper(gym.Env):
    """
    Transforme l'env 2-agents en env 1-agent pour rlgym-ppo.
    Blue = entraîné par PPO.
    Orange = contrôlé par l'heuristique (ne participe pas aux gradients).
    """

    def __init__(self, inner_env):
        self._inner = inner_env          # MultiDiscreteGymWrapper (2 agents)
        n_obs = inner_env.observation_space.shape[0]
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_obs,), dtype=np.float32
        )
        self.action_space = inner_env.action_space
        self._blue_idx   = 0
        self._orange_idx = 1

    def _find_indices(self):
        """Identifie quel index correspond à blue/orange après reset."""
        state = self._inner.rlgym_env.state
        for idx, agent_id in self._inner.agent_map.items():
            if state.cars[agent_id].is_blue:
                self._blue_idx   = idx
            else:
                self._orange_idx = idx

    def reset(self):
        obs_all = self._inner.reset()
        self._find_indices()
        return obs_all[self._blue_idx]

    def step(self, blue_action):
        # Calcul de l'action orange AVANT le step (sur l'état courant)
        state     = self._inner.rlgym_env.state
        orange_id = self._inner.agent_map[self._orange_idx]
        opp_action = _heuristic_action(state, orange_id)

        # rlgym-ppo peut envoyer blue_action en (1,8) ou (8,) — normaliser
        blue_flat = np.asarray(blue_action).flatten()   # toujours (8,)

        # Construction du tableau d'actions pour les 2 agents
        n_agents = len(self._inner.agent_map)
        actions  = np.zeros((n_agents, len(blue_flat)), dtype=np.float32)
        actions[self._blue_idx]   = blue_flat
        actions[self._orange_idx] = opp_action

        obs_all, rews, done, truncated, info = self._inner.step(actions)

        return obs_all[self._blue_idx], rews[self._blue_idx], done, truncated, info

    def close(self):
        self._inner.close()


# ---------------------------------------------------------------------------
# Environment Builder
# ---------------------------------------------------------------------------

def build_rlgym_v2_env():
    action_parser = RepeatAction(MultiDiscreteActionParser(), repeats=8)

    termination_condition = GoalCondition()
    truncation_condition  = AnyCondition(
        NoTouchTimeoutCondition(timeout_seconds=30),
        TimeoutCondition(timeout_seconds=300),
    )

    obs_builder = ExtendedObs(
        zero_padding=1,
        pos_coef=np.asarray([
            1 / common_values.SIDE_WALL_X,
            1 / common_values.BACK_NET_Y,
            1 / common_values.CEILING_Z,
        ]),
        ang_coef     = 1 / np.pi,
        lin_vel_coef = 1 / common_values.CAR_MAX_SPEED,
        ang_vel_coef = 1 / common_values.CAR_MAX_ANG_VEL,
        boost_coef   = 1 / 100.0,
    )

    state_mutator = MutatorSequence(
        FixedTeamSizeMutator(blue_size=1, orange_size=1),
        KickoffMutator(),
    )

    rlgym_env = RLGym(
        state_mutator    = state_mutator,
        obs_builder      = obs_builder,
        action_parser    = action_parser,
        reward_fn        = CurriculumCombinedReward(n_proc=N_PROC),
        termination_cond = termination_condition,
        truncation_cond  = truncation_condition,
        transition_engine= RocketSimEngine(),
    )

    return HeuristicOpponentWrapper(MultiDiscreteGymWrapper(rlgym_env))


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
        checkpoints_save_folder = "data/checkpoints/prl-run",
        add_unix_timestamp      = False,
        n_checkpoints_to_keep   = 10,
        ppo_batch_size          = 60_000,
        policy_layer_sizes      = [512, 512],
        critic_layer_sizes      = [512, 512],
        ts_per_iteration        = 120_000,  # 2× : plus de signal sparse par update
        exp_buffer_size         = 360_000,  # ts_per_iteration × 3
        ppo_minibatch_size      = 12_000,   # ratio constant
        ppo_ent_coef            = 0.0005,
        policy_lr               = LR_INITIAL,
        critic_lr               = LR_INITIAL,
        ppo_epochs              = 2,        # 3→2 : évite la dégradation de représentation
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
