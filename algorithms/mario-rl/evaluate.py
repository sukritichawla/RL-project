
import numpy as np
import torch

from mario_env import MarioEnv
from dqn_agent import DQN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "checkpoints/best_mario_dqn.pth"
EPISODES = 5


def evaluate():
    env = MarioEnv()
    model = DQN(num_actions=env.action_space.n).to(DEVICE)

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    )
    model.eval()

    rewards = []

    try:
        for episode in range(1, EPISODES + 1):
            state = env.reset()
            total_reward = 0
            done = False

            while not done:
                state_tensor = torch.as_tensor(
                    np.array(state),
                    dtype=torch.float32,
                    device=DEVICE
                ).unsqueeze(0)

                with torch.no_grad():
                    action = model(state_tensor).argmax(dim=1).item()

                state, reward, done, info = env.step(action)
                total_reward += reward

            rewards.append(total_reward)
            print(
                f"Evaluation episode {episode}/{EPISODES} | "
                f"Reward: {total_reward:.1f}"
            )

    finally:
        env.close()

    print(f"\nAverage evaluation reward: {np.mean(rewards):.2f}")


if __name__ == "__main__":
    evaluate()
