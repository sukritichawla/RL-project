import torch

from algorithms.deep.dqn_agent import DQNAgent
from trainers.evaluate import Evaluator


agent = DQNAgent(
    state_size=6,
    action_size=6
)

agent.online_network.load_state_dict(
    torch.load(r"C:\Users\Noureen\Documents\B.Tech\6thSem\RL\RL_\RL-project\models\double_dqn_model.pth")
)

agent.epsilon = 0.0

evaluator = Evaluator(agent)

print("\n===== Double DQN EVALUATION =====")

evaluator.evaluate(episodes=20)