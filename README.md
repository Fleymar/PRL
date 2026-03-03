# PRL – Rocket League School Project

This repository contains a school project called **PRL**, focused on building and training Rocket League bots using **RLBot** and **RLGym / PPO**.

- **Opponent bot** (hard-coded logic) lives in the `Opponent/` folder.
- **Reinforcement learning (PPO) training setup** lives in the `AI/` folder.

The goal is to:

1. Have a reliable hard-coded opponent bot that can play full games.
2. Train an AI bot with PPO on a simulated Rocket League environment.

---

## Project structure

- `Opponent/`

  - `rlbot.toml` – RLBot match configuration for the opponent project.
  - `run.py`, `run_only.py` – helper scripts to launch RLBot.
  - `src/bot.py` – main hard-coded opponent bot (`MyBot`).
  - `src/bot.toml`, `src/loadout.toml` – agent configuration and car loadout.
  - `src/util/` – shared helper modules (physics, rendering, sequences, etc.).
  - `requirements.txt` – Python dependencies for the RLBot opponent.
- `AI/`

  - [ ] `train_ppo.py` – PPO training script using `rlgym` and `rlgym_ppo`.
    - [ ] Defines `build_rlgym_v2_env()` (action parser, reward, obs builder, mutators, RocketSim engine).
    - [ ] Creates a `Learner` and runs a basic PPO training loop.
- `.gitignore` – excludes `venv`, training data and other generated files from git.

---

## Prerequisites

- Windows with **Rocket League** installed (Steam / Epic).
- **Python 3.11+** (what the project currently uses).
- Git.

---

## Setup

From the project root:

```bash
python -m venv venv
venv\Scripts\activate  # on PowerShell / cmd
# or
source venv/Scripts/activate  # in Git Bash with the right path

# Install opponent bot requirements
pip install -r Opponent/requirements.txt

# Install RL training stack
pip install "rlgym[rl-sim]"
pip install git+https://github.com/AechPro/rlgym-ppo
pip install torch  # or a CUDA build from pytorch.org if you have a GPU
```

---

## Running the opponent bot

There are two common ways:

1. **Via RLBot GUI**

   - Open `Opponent/rlbot.toml` in the RLBot GUI.
   - Click **Run** to start a match with the opponent bot.
2. **Via script** (depending on how `run.py` is set up):

```bash
venv\Scripts\activate
python Opponent/run.py
```

The hard-coded bot lives in `Opponent/src/bot.py` and uses the utilities in `Opponent/src/util/`.

---

## Running PPO training (AI)

From the project root:

```bash
venv\Scripts\activate
python AI/train_ppo.py
```

This will:

- Build an `RLGym` v2 environment (`build_rlgym_v2_env`).
- Wrap it with `RLGymV2GymWrapper`.
- Start a **PPO** learner with a small batch size (good for quick tests).
- Create a `data/` folder inside `AI/` containing checkpoints and logs.

The first training run is mainly to verify:

- The **environment** is correctly configured.
- The **GPU/CPU** is used (depending on your PyTorch install, whether or not you are using CUDA).
- The training loop runs without crashing.

---

## Notes

- The virtual environment `venv/` and large generated files are intentionally ignored by git.
- The current AI part, is unavailable. It will be added later on in the project, for now we just have made sure that a PPO algorithm would run smoothly.
- This codebase is meant as a teaching / experimentation project, not a polished production bot.
- Feel free to tweak hyperparameters in `AI/train_ppo.py` once everything is confirmed working.
