"""
PPO Bot - Intégration RLBot du modèle entraîné avec rlgym-ppo (MultiDiscrete).

Reproduit fidèlement ExtendedObs (101 floats pour 1v1) :
  9  balle  (pos, lin_vel, ang_vel)
  34 timers boost pads
  9  vars partiellement observables du bot
  20 obs voiture self
  20 obs voiture adversaire
  9  vecteurs relatifs (ball→their_goal, ball→own_goal, car→ball)
"""

import math
from pathlib import Path

import numpy as np
import torch

try:
    from typing import override
except ImportError:
    def override(method):
        return method

from rlbot.flat import ControllerState, GamePacket
from rlbot.managers import Bot
from rlbot_flatbuffers import AirState

# ---------------------------------------------------------------------------
# Constantes de normalisation — doivent correspondre exactement à train_ppo.py
# ---------------------------------------------------------------------------
from rlgym.rocket_league import common_values

POS_COEF = np.array(
    [1.0 / common_values.SIDE_WALL_X,
     1.0 / common_values.BACK_NET_Y,
     1.0 / common_values.CEILING_Z],
    dtype=np.float32,
)
LIN_VEL_COEF  = np.float32(1.0 / common_values.CAR_MAX_SPEED)
ANG_VEL_COEF  = np.float32(1.0 / common_values.CAR_MAX_ANG_VEL)
BOOST_COEF    = np.float32(1.0 / 100.0)
PAD_TIMER_COEF = np.float32(1.0 / 10.0)   # default dans DefaultObs

# Inversion pour l'équipe orange : multiplier tous les vecteurs par [-1,-1,1]
INV = np.array([-1.0, -1.0, 1.0], dtype=np.float32)

# Nombre de frames de répétition (identique à action_repeat dans train_ppo.py)
ACTION_REPEAT = 8

# Délai max avant qu'un double-saut/flip expire (DOUBLEJUMP_MAX_DELAY de rlgym)
DOUBLEJUMP_MAX_DELAY = 1.25

# Bins MultiDiscrete — identiques à _BINS dans train_ppo.py
# [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _euler_to_forward_up(pitch: float, yaw: float, roll: float):
    """Retourne (forward, up) — vecteurs unités (3,) — depuis les angles d'Euler."""
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw),   math.sin(yaw)
    cr, sr = math.cos(roll),  math.sin(roll)

    forward = np.array([cp * cy, cp * sy, sp], dtype=np.float32)
    up = np.array([cy * sp * cr + sr * sy,
                   sy * sp * cr - sr * cy,
                   cp * cr], dtype=np.float32)

    return forward, up


def _car_obs(player, invert: bool) -> np.ndarray:
    """20 floats par voiture."""
    ph  = player.physics
    pos = np.array([ph.location.x,         ph.location.y,         ph.location.z],         dtype=np.float32)
    vel = np.array([ph.velocity.x,         ph.velocity.y,         ph.velocity.z],         dtype=np.float32)
    ang = np.array([ph.angular_velocity.x, ph.angular_velocity.y, ph.angular_velocity.z], dtype=np.float32)
    fwd, up = _euler_to_forward_up(ph.rotation.pitch, ph.rotation.yaw, ph.rotation.roll)

    if invert:
        pos *= INV
        vel *= INV
        ang *= INV
        fwd *= INV
        up  *= INV

    boost_n      = float(player.boost) * BOOST_COEF
    demo_timer   = float(player.demolished_timeout)
    on_ground    = float(player.air_state == AirState.OnGround)
    is_boosting  = float(player.last_input.boost)
    is_supersonic = float(player.is_supersonic)

    return np.concatenate([
        pos * POS_COEF,
        fwd,
        up,
        vel * LIN_VEL_COEF,
        ang * ANG_VEL_COEF,
        [boost_n, demo_timer, on_ground, is_boosting, is_supersonic],
    ])


def _partial_obs(player, air_time_since_jump: float) -> np.ndarray:
    """9 floats — variables partiellement observables du bot."""
    air_state       = player.air_state
    is_jumping      = (air_state == AirState.Jumping)
    is_flipping     = (air_state == AirState.Dodging)
    has_jumped      = player.has_jumped
    has_flipped     = player.has_dodged
    has_double_jumped = player.has_double_jumped
    is_holding_jump = bool(player.last_input.jump)
    on_ground       = (air_state == AirState.OnGround)

    has_flip = (not has_double_jumped
                and not has_flipped
                and air_time_since_jump < DOUBLEJUMP_MAX_DELAY)
    can_flip = (not on_ground and not is_holding_jump and has_flip)

    return np.array([
        float(is_holding_jump),
        float(player.last_input.handbrake),
        float(has_jumped),
        float(is_jumping),
        float(has_flipped),
        float(is_flipping),
        float(has_double_jumped),
        float(can_flip),
        air_time_since_jump,
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# Bot RLBot
# ---------------------------------------------------------------------------

class PPOBot(Bot):
    """Bot RLBot piloté par la policy MultiDiscrete PPO entraînée avec rlgym-ppo."""

    policy       = None
    _tick        : int = 0
    _controls    : ControllerState = None
    _air_time: dict = None

    @override
    def initialize(self):
        self._load_policy()
        self._tick     = ACTION_REPEAT
        self._controls = ControllerState()
        self._air_time = {}

    # ------------------------------------------------------------------
    # Chargement du modèle
    # ------------------------------------------------------------------

    def _load_policy(self):
        checkpoint_root = Path(__file__).resolve().parent / "data" / "checkpoints"
        run_dirs = sorted(p for p in checkpoint_root.iterdir() if p.is_dir())
        if not run_dirs:
            raise FileNotFoundError(f"Aucun run trouvé dans {checkpoint_root}")

        step_dirs = sorted(
            (p for p in run_dirs[-1].iterdir() if p.is_dir() and p.name.isdigit()),
            key=lambda p: int(p.name),
        )
        if not step_dirs:
            raise FileNotFoundError(f"Aucun checkpoint dans {run_dirs[-1]}")

        policy_path = step_dirs[-1] / "PPO_POLICY.pt"
        print(f"[PPOBot] Chargement : {policy_path}")

        state_dict = torch.load(policy_path, map_location="cpu")

        from rlgym_ppo.ppo import MultiDiscreteFF
        keys     = list(state_dict.keys())
        obs_size = state_dict[keys[0]].shape[1]   # 1ère couche → (hidden, obs_size)
        print(f"[PPOBot] obs_size={obs_size}")

        self.policy = MultiDiscreteFF(obs_size, [512, 512], torch.device("cpu"))
        self.policy.load_state_dict(state_dict)
        self.policy.eval()
        print(f"[PPOBot] Policy MultiDiscrete prête ({step_dirs[-1].name} timesteps)")

    # ------------------------------------------------------------------
    # Boucle principale
    # ------------------------------------------------------------------

    @override
    def get_output(self, packet: GamePacket) -> ControllerState:
        if len(packet.balls) == 0:
            return ControllerState()

        self._update_air_time(packet)

        self._tick += 1
        if self._tick < ACTION_REPEAT:
            return self._controls
        self._tick = 0

        obs   = self._build_obs(packet)
        obs_t = torch.FloatTensor(obs).unsqueeze(0)

        with torch.no_grad():
            action_indices, _ = self.policy.get_action(obs_t, deterministic=True)
            # action_indices shape : (8, 1) numpy array
            # Chaque ligne = index discret pour une dimension d'action

        # Convertir les indices en valeurs réelles via _BINS
        cs           = ControllerState()
        cs.throttle  = float(_BINS[0][int(action_indices[0])])
        cs.steer     = float(_BINS[1][int(action_indices[1])])
        cs.pitch     = float(_BINS[2][int(action_indices[2])])
        cs.yaw       = float(_BINS[3][int(action_indices[3])])
        cs.roll      = float(_BINS[4][int(action_indices[4])])
        cs.jump      = bool(_BINS[5][int(action_indices[5])])
        cs.boost     = bool(_BINS[6][int(action_indices[6])])
        cs.handbrake = bool(_BINS[7][int(action_indices[7])])

        self._controls = cs
        return cs

    # ------------------------------------------------------------------
    # Suivi air_time_since_jump
    # ------------------------------------------------------------------

    def _update_air_time(self, packet: GamePacket):
        DT = 1.0 / 60.0
        for i, player in enumerate(packet.players):
            on_ground  = (player.air_state == AirState.OnGround)
            is_jumping = (player.air_state == AirState.Jumping)
            if on_ground or not player.has_jumped or is_jumping:
                self._air_time[i] = 0.0
            else:
                self._air_time[i] = self._air_time.get(i, 0.0) + DT

    # ------------------------------------------------------------------
    # Construction de l'observation (reproduit ExtendedObs de train_ppo.py)
    # ------------------------------------------------------------------

    def _build_obs(self, packet: GamePacket) -> np.ndarray:
        me     = packet.players[self.index]
        invert = (me.team == 1)   # 1 = orange

        # ---- Balle (9 floats) ----
        b   = packet.balls[0].physics
        b_pos = np.array([b.location.x, b.location.y, b.location.z], dtype=np.float32)
        b_vel = np.array([b.velocity.x, b.velocity.y, b.velocity.z], dtype=np.float32)
        b_ang = np.array([b.angular_velocity.x, b.angular_velocity.y, b.angular_velocity.z], dtype=np.float32)
        if invert:
            b_pos *= INV; b_vel *= INV; b_ang *= INV
        ball_part = np.concatenate([b_pos * POS_COEF, b_vel * LIN_VEL_COEF, b_ang * ANG_VEL_COEF])

        # ---- Timers boost pads (34 floats) ----
        pad_timers = np.array(
            [p.timer for p in packet.boost_pads], dtype=np.float32
        )
        if invert:
            pad_timers = pad_timers[::-1].copy()
        pad_part = pad_timers * PAD_TIMER_COEF

        # ---- Vars partiellement observables du bot (9 floats) ----
        partial_part = _partial_obs(me, self._air_time.get(self.index, 0.0))

        # ---- Obs voitures (20 floats chacune) ----
        self_part = _car_obs(me, invert)
        opp_parts = [
            _car_obs(packet.players[i], invert)
            for i in range(len(packet.players))
            if i != self.index
        ]

        # ---- Vecteurs relatifs (9 floats) — identique à ExtendedObs dans train_ppo.py ----
        ph_me = me.physics
        car_pos = np.array([ph_me.location.x, ph_me.location.y, ph_me.location.z], dtype=np.float32)
        if invert:
            car_pos = car_pos * INV

        from rlgym.rocket_league.common_values import BACK_NET_Y
        own_goal   = np.array([0.0,  BACK_NET_Y, 0.0], dtype=np.float32)
        their_goal = np.array([0.0, -BACK_NET_Y, 0.0], dtype=np.float32)

        ball_pos_norm = b_pos  # déjà inversé si orange

        def unit(v):
            n = np.linalg.norm(v)
            return v / n if n > 1e-6 else v

        ball_to_their_goal = unit(their_goal - ball_pos_norm)
        ball_to_own_goal   = unit(own_goal   - ball_pos_norm)
        car_to_ball        = unit(ball_pos_norm - car_pos)

        extra = np.concatenate([ball_to_their_goal, ball_to_own_goal, car_to_ball])

        obs = np.concatenate([ball_part, pad_part, partial_part, self_part] + opp_parts + [extra])
        return obs.astype(np.float32)


if __name__ == "__main__":
    PPOBot("rlbot_community/prl_ppo").run()
