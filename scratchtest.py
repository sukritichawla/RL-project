import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "environment"))

from environment.water_env import WaterEnvironment
from algorithms.classical.monte_carlo import MonteCarloAgent, MonteCarloConfig
from trainers.train import Trainer
from trainers.evaluate import Evaluator
from config.training_config import EPISODES

# --- Train ---
env = WaterEnvironment()
state = env.reset()

agent = MonteCarloAgent(state_size=len(state), action_size=6, config=MonteCarloConfig(verbose=True))
trainer = Trainer(agent, episodes=EPISODES)
history = trainer.train()

print("\nFinal training stats:", agent.get_q_table_stats())
print("Avg reward, last 50 training episodes:", sum(history[-50:]) / 50)

# --- Evaluate ---
print("\nRunning evaluation...")
evaluator = Evaluator(agent)
eval_avg = evaluator.evaluate(episodes=20)

# --- Save trained model ---
agent.save("models/classical/monte_carlo/monte_carlo_qtable.pkl")
print("\nModel saved.")