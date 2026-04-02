import os
from pathlib import Path
from typing import Any, Dict, List

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
from rlgym.rocket_league.reward_functions import GoalReward, TouchReward
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
TOTAL_TIMESTEPS  = 100_000_000
N_PROC           = 12

# Phases du curriculum (en timesteps globaux estimés)
PHASE1_END = 20_000_000   # 0  → 20M : apprendre à toucher la balle
PHASE2_END = 60_000_000   # 20M→ 60M : orienter la balle vers le but
                           # 60M→100M : marquer / défendre

# Learning rate schedule (décroissance linéaire)
LR_INITIAL = 5e-4
LR_FINAL   = 5e-5

# ---------------------------------------------------------------------------
# TensorBoard + LR Scheduler
# ---------------------------------------------------------------------------
tb_writer = SummaryWriter(log_dir="data/tensorboard", flush_secs=10)
_learner  = None   # initialisé dans main(), utilisé par le scheduler

_original_report = _reporting.report_metrics

def _tb_report_metrics(loggable_metrics, debug_metrics, wandb_run=None):
    _original_report(loggable_metrics, debug_metrics, wandb_run)

    step = loggable_metrics.get("Cumulative Timesteps", 0)

    # --- LR scheduler (décroissance linéaire) ---
    if _learner is not None:
        t      = min(step / TOTAL_TIMESTEPS, 1.0)
        new_lr = LR_INITIAL + (LR_FINAL - LR_INITIAL) * t
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
    """
    Étend DefaultObs (92 floats) avec 9 floats :
      ball → but adverse (3), ball → propre cage (3), car → ball (3)
    Total : 101 floats
    """

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
# 8 têtes indépendantes :
# [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]
# Le bot apprend "Accélérer" et "Tourner"   distinctement

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
_BIN_SIZES = [len(b) for b in _BINS]   # [3,3,3,3,3,2,2,2]


class MultiDiscreteActionParser(ActionParser):
    """
    8 dimensions d'action indépendantes.
    Chaque dimension est un choix discret parmi 2 ou 3 valeurs.
    """

    def get_action_space(self, agent: AgentID):
        return 'multi_discrete', _BIN_SIZES

    def reset(self, agents, initial_state, shared_info) -> None:
        pass

    def parse_actions(self, actions: Dict[AgentID, np.ndarray],
                      state: GameState,
                      shared_info: Dict[str, Any]) -> Dict[AgentID, np.ndarray]:
        parsed = {}
        for agent, action in actions.items():
            # action = array d'indices, une par dimension
            parsed[agent] = np.array(
                [_BINS[i][int(action[i])] for i in range(len(_BINS))],
                dtype=np.float32,
            )
        return parsed


class MultiDiscreteGymWrapper(RLGymV2GymWrapper):
    """
    Étend RLGymV2GymWrapper pour supporter gym.spaces.MultiDiscrete.
    Cela déclenche automatiquement MultiDiscreteFF dans rlgym-ppo.
    """

    def __init__(self, rlgym_env):
        super().__init__(rlgym_env)
        act_space = list(rlgym_env.action_spaces.values())[0][1]
        if isinstance(act_space, (list, tuple)):
            self.action_space = gym.spaces.MultiDiscrete(act_space)


# ---------------------------------------------------------------------------
# Reward Functions
# ---------------------------------------------------------------------------

class VelocityPlayerToBallReward(RewardFunction[AgentID, GameState, float]):
    def reset(self, _agents, _initial_state, _shared_info) -> None:
        pass

    def get_rewards(self, agents, state, _is_terminated, _is_truncated, _shared_info):
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
            rewards[agent] = vel_toward / common_values.CAR_MAX_SPEED
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


class DefensivePenaltyReward(RewardFunction[AgentID, GameState, float]):
    """
    Petite pénalité continue quand la balle est dans notre moitié de terrain.
    Encourage à garder la balle dans le camp adverse.
    Poids très faible (~0.001) pour éviter le "suicide bot" trap.
    """
    def reset(self, agents, initial_state, shared_info) -> None:
        pass

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        rewards = {}
        ball_y = state.ball.position[1]
        for agent in agents:
            car = state.cars[agent]
            # Balle dans notre camp = pénalité -1.0 (multipliée par un poids tiny)
            if car.team_num == BLUE_TEAM:
                in_our_half = ball_y < 0.0
            else:
                in_our_half = ball_y > 0.0
            rewards[agent] = -1.0 if in_our_half else 0.0
        return rewards


class HighVelocitySaveReward(RewardFunction[AgentID, GameState, float]):
    """
    Récompense le bot pour avoir touché la balle alors qu'elle fonçait
    rapidement vers notre cage. Encourage les arrêts défensifs.
    """
    SPEED_THRESHOLD = 800.0   # UU/s minimum vers notre cage pour déclencher

    def reset(self, agents, initial_state, shared_info) -> None:
        pass

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        rewards = {}
        ball_vel = state.ball.linear_velocity
        for agent in agents:
            car = state.cars[agent]
            # Composante de vitesse de la balle vers notre cage
            if car.team_num == BLUE_TEAM:
                vel_toward_our_goal = -ball_vel[1]   # vers Y négatif = vers notre cage
            else:
                vel_toward_our_goal =  ball_vel[1]   # vers Y positif = vers notre cage

            if vel_toward_our_goal > self.SPEED_THRESHOLD and car.ball_touches > 0:
                rewards[agent] = vel_toward_our_goal / common_values.BALL_MAX_SPEED
            else:
                rewards[agent] = 0.0
        return rewards


# ---------------------------------------------------------------------------
# Curriculum Combined Reward
# ---------------------------------------------------------------------------

def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * max(0.0, min(1.0, t))


class CurriculumCombinedReward(RewardFunction[AgentID, GameState, float]):
    """
    Reward combinée avec poids évoluant automatiquement selon les phases :

    Phase 1 (0→20M)  : apprendre à bouger et toucher la balle
    Phase 2 (20→60M) : orienter la balle vers le but
    Phase 3 (60→100M): marquer / défendre

    Chaque instance de reward tourne sur un worker distinct.
    On estime les timesteps globaux via : steps_locaux × N_PROC.
    """

    def __init__(self, n_proc: int = N_PROC):
        self._n_proc        = n_proc
        self._local_steps   = 0
        self._goal          = GoalReward()
        self._ball_to_goal  = VelocityBallToGoalReward()
        self._player_to_ball = VelocityPlayerToBallReward()
        self._touch         = TouchReward()
        self._defensive     = DefensivePenaltyReward()
        self._save          = HighVelocitySaveReward()

    @property
    def _global_steps(self) -> int:
        return self._local_steps * self._n_proc

    def _get_weights(self) -> dict:
        gs = self._global_steps

        if gs <= PHASE1_END:
            # Poids FIXES — le bot apprend à bouger et toucher sans que la cible bouge
            return dict(
                goal          = 5.0,
                ball_to_goal  = 0.5,
                player_to_ball= 2.0,
                touch         = 3.0,
                defensive     = 0.0,
                save          = 0.0,
            )
        elif gs <= PHASE2_END:
            # Transition progressive vers comportement orienté but
            t = (gs - PHASE1_END) / (PHASE2_END - PHASE1_END)
            return dict(
                goal          = _lerp( 5.0, 15.0, t),
                ball_to_goal  = _lerp( 0.5,  4.0, t),
                player_to_ball= _lerp( 2.0,  0.5, t),
                touch         = _lerp( 3.0,  1.0, t),
                defensive     = _lerp( 0.0,  0.002, t),
                save          = _lerp( 0.0,  3.0, t),
            )
        else:
            # Focus marquer / défendre
            t = min((gs - PHASE2_END) / (TOTAL_TIMESTEPS - PHASE2_END), 1.0)
            return dict(
                goal          = _lerp(15.0, 25.0, t),
                ball_to_goal  = _lerp( 4.0,  2.0, t),
                player_to_ball= 0.3,
                touch         = 0.5,
                defensive     = _lerp( 0.002, 0.004, t),
                save          = _lerp( 3.0,  5.0, t),
            )

    def reset(self, agents, initial_state, shared_info) -> None:
        self._goal.reset(agents, initial_state, shared_info)
        self._ball_to_goal.reset(agents, initial_state, shared_info)
        self._player_to_ball.reset(agents, initial_state, shared_info)
        self._touch.reset(agents, initial_state, shared_info)
        self._defensive.reset(agents, initial_state, shared_info)
        self._save.reset(agents, initial_state, shared_info)

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        self._local_steps += 1
        w = self._get_weights()

        r_goal   = self._goal.get_rewards(agents, state, is_terminated, is_truncated, shared_info)
        r_btg    = self._ball_to_goal.get_rewards(agents, state, is_terminated, is_truncated, shared_info)
        r_ptb    = self._player_to_ball.get_rewards(agents, state, is_terminated, is_truncated, shared_info)
        r_touch  = self._touch.get_rewards(agents, state, is_terminated, is_truncated, shared_info)
        r_def    = self._defensive.get_rewards(agents, state, is_terminated, is_truncated, shared_info)
        r_save   = self._save.get_rewards(agents, state, is_terminated, is_truncated, shared_info)

        return {
            agent: (
                w['goal']          * r_goal[agent]  +
                w['ball_to_goal']  * r_btg[agent]   +
                w['player_to_ball']* r_ptb[agent]   +
                w['touch']         * r_touch[agent] +
                w['defensive']     * r_def[agent]   +
                w['save']          * r_save[agent]
            )
            for agent in agents
        }


# ---------------------------------------------------------------------------
# Environment Builder
# ---------------------------------------------------------------------------

def build_rlgym_v2_env():
    action_repeat          = 8
    no_touch_timeout_secs  = 30
    game_timeout_secs      = 300

    action_parser = RepeatAction(MultiDiscreteActionParser(), repeats=action_repeat)

    termination_condition = GoalCondition()
    truncation_condition  = AnyCondition(
        NoTouchTimeoutCondition(timeout_seconds=no_touch_timeout_secs),
        TimeoutCondition(timeout_seconds=game_timeout_secs),
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
        state_mutator      = state_mutator,
        obs_builder        = obs_builder,
        action_parser      = action_parser,
        reward_fn          = CurriculumCombinedReward(n_proc=N_PROC),
        termination_cond   = termination_condition,
        truncation_cond    = truncation_condition,
        transition_engine  = RocketSimEngine(),
    )

    return MultiDiscreteGymWrapper(rlgym_env)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main():
    global _learner

    min_inference_size = max(1, int(round(N_PROC * 0.9)))

    _learner = Learner(
        build_rlgym_v2_env,
        n_proc             = N_PROC,
        min_inference_size = min_inference_size,
        metrics_logger     = None,
        device             = "cuda",
        ppo_batch_size     = 60_000,
        policy_layer_sizes = [512, 512],
        critic_layer_sizes = [512, 512],
        ts_per_iteration   = 60_000,
        exp_buffer_size    = 180_000,
        ppo_minibatch_size = 6_000,
        ppo_ent_coef       = 0.005,
        policy_lr          = LR_INITIAL,
        critic_lr          = LR_INITIAL,
        ppo_epochs         = 3,
        standardize_returns= True,
        standardize_obs    = False,
        save_every_ts      = 500_000,
        timestep_limit     = TOTAL_TIMESTEPS,
        log_to_wandb       = False,
    )

    _learner.learn()
    tb_writer.close()


if __name__ == "__main__":
    main()
