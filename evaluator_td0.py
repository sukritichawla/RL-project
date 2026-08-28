from algorithms.classical.temporal_difference.td0_agent import TD0Agent
from trainers.evaluate import Evaluator


agent = TD0Agent(
    state_size=6,
    action_size=6
)

# No exploration during evaluation
agent.epsilon = 0.0

evaluator = Evaluator(agent)

print("\n===== TD(0) EVALUATION =====")

evaluator.evaluate(episodes=20)