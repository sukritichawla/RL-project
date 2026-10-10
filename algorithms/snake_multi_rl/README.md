# Snake Game with Multiple Reinforcement Learning Algorithms

A runnable educational comparison of four agents trained on the same Snake game:

- **Q-Learning**: tabular off-policy TD control
- **SARSA**: tabular on-policy TD control
- **DQN**: deep Q-learning with replay memory and a target network
- **Double DQN**: online network selects the next action; target network evaluates it

The RL algorithms and update logic are implemented in this project. NumPy is used for tabular data and PyTorch is used as the neural-network/tensor library for DQN agents.

## Features

- Shared 12 × 12 Snake environment with 3 relative actions
- Train all four agents and save models
- Evaluate each model on fresh episodes
- CSV metrics and learning/evaluation plots
- Pygame viewer for trained agents and manual play
- Console comparison dashboard

## Install

Python 3.10 or 3.11 is recommended. Open a terminal in this folder.

### Windows PowerShell
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
If PowerShell blocks activation, use Command Prompt instead:
```bat
py -3.11 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS/Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run everything

```bash
python train.py
python evaluate.py
python dashboard.py
```

Training runs 1,200 episodes per agent by default and can take a while, especially on CPU. For a quick smoke test, set `EPISODES = 10` in `config.py`, train and evaluate, then restore the default for experiments.

## Watch the game

After training, run any agent:
```bash
python play.py --agent q_learning
python play.py --agent sarsa
python play.py --agent dqn
python play.py --agent double_dqn
```
Or play yourself:
```bash
python play.py --agent human
```
Arrow keys steer, SPACE pauses, R resets and ESC exits.

## Files created

- `models/q_learning.npz`, `models/sarsa.npz`
- `models/dqn.pt`, `models/double_dqn.pt`
- `results/training_metrics.csv`
- `results/evaluation_metrics.csv`
- `results/learning_curves.png`
- `results/evaluation_boxplot.png`
- `results/evaluation_comparison.png`

## Environment design

Observation is an 11-element binary feature vector:
1. Danger straight, right and left
2. Current direction (up, right, down, left)
3. Food relative to the head (left, right, up, down)

Tabular algorithms pack those bits into an integer key. Deep agents receive the vector directly.

Actions: `0 = straight`, `1 = turn right`, `2 = turn left`.

Rewards: `+10` for food, `-0.01` per ordinary step, `-10` for collision and `-2` at the step limit.

## Algorithm update rules

Q-Learning:
`Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') − Q(s,a)]`

SARSA:
`Q(s,a) ← Q(s,a) + α [r + γ Q(s',a') − Q(s,a)]`

DQN uses a replay buffer and target network. Double DQN separates next-action selection (online network) from next-action valuation (target network), reducing maximization bias.

## Caveats

This compact state omits most of the board, so agents cannot reliably plan around the entire snake body. Results are an educational baseline, not a universal algorithm ranking. Run multiple random seeds and report means/standard deviations before making strong comparisons. No real-world actions or external services are used.
