# PRL — Rocket League RL Bot

Reinforcement learning bot for Rocket League, trained with PPO using [rlgym-ppo](https://github.com/AechPro/rlgym-ppo) and [RocketSim](https://github.com/ZealanL/RocketSim).

## Project structure

```text
AI/
  train_ppo.py   — PPO training script
  bot.py         — RLBot v5 inference bot
  bot.toml       — Bot configuration
  rlbot.toml     — Match configuration
Opponent/
  src/bot.py     — Hard-coded rule-based opponent
```

## Architecture

### Observation — 101 floats

| Block | Size | Content |
| --- | --- | --- |
| Ball | 9 | position, linear velocity, angular velocity |
| Boost pads | 34 | recharge timer per pad |
| Car (self) | 20 | position, orientation, velocities, boost, state |
| Car (opponent) | 20 | same for opponent |
| Relative vectors | 9 | ball→their goal, ball→own goal, car→ball |

Values normalized by field dimensions and max speeds. Orange team observations are inverted on `[-1, -1, 1]`.

### Action space — 90 discrete actions

Pruned lookup table combining ground actions (throttle × steer × boost × handbrake) and aerial actions (pitch × yaw × roll × jump × boost). Each action repeats for 8 physics frames (~133ms).

### Reward

Delta-based rewards to prevent reward farming:

- Delta player quality (Liu distance + alignment toward goal)
- Delta state quality (ball toward opponent goal)
- Touch height with wall distance factor
- Touch acceleration
- Flip reset
- Boost management (sqrt-scaled gain, ground loss penalty)
- Angular velocity encouragement
- Demo events, goal reward

Team spirit blending (0.6) with opponent punishment (1.0).

### Network

`101 → 1024 → 1024 → 90` with ReLU, trained with Adam.

### Self-play

Population self-play: the orange player uses a randomly sampled checkpoint from the last 10 saves, swapped every 3M steps.

## Prerequisites

- Windows with Rocket League installed (Steam)
- Python 3.11+
- CUDA-capable GPU (recommended)

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

For GPU training (recommended), install PyTorch with CUDA 12.8 instead:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

## Training

```bash
venv\Scripts\activate
python AI/train_ppo.py
```

Checkpoints are saved to `AI/data/checkpoints/` every 500k steps.

## Running the bot

Add `AI/bot.toml` in the RLBot GUI. The bot automatically loads the latest checkpoint.

## Running the opponent

Add `Opponent/src/bot.toml` in the RLBot GUI.
