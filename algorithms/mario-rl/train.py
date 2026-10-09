
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from mario_env import MarioEnv
from dqn_agent import DQN
from replay_buffer import ReplayBuffer


# Training settings

EPISODES = 50
BATCH_SIZE = 16
GAMMA = 0.99
LEARNING_RATE = 1e-4
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 20000
TARGET_UPDATE = 1000
MEMORY_SIZE = 10000
WARMUP_STEPS = 500


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def choose_action(model, state, epsilon, num_actions):
    if random.random() < epsilon:
        return random.randrange(num_actions)

    state_tensor = torch.as_tensor(
        np.array(state), dtype=torch.float32, device=DEVICE
    ).unsqueeze(0)

    with torch.no_grad():
        q_values = model(state_tensor)

    return q_values.argmax(dim=1).item()


def train():
    env = MarioEnv()
    num_actions = env.action_space.n

    policy_net = DQN(num_actions=num_actions).to(DEVICE)
    target_net = DQN(num_actions=num_actions).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    memory = ReplayBuffer(MEMORY_SIZE)

    os.makedirs("checkpoints", exist_ok=True)

    total_steps = 0
    best_reward = float("-inf")

    try:
        for episode in range(1, EPISODES + 1):
            state = env.reset()
            episode_reward = 0.0
            done = False

            while not done:
                epsilon = max(
                    EPSILON_END,
                    EPSILON_START - total_steps / EPSILON_DECAY
                    * (EPSILON_START - EPSILON_END)
                )

                action = choose_action(
                    policy_net, state, epsilon, num_actions
                )

                next_state, reward, done, info = env.step(action)

                memory.add(
                    state, action, reward, next_state, done
                )

                state = next_state
                episode_reward += reward
                total_steps += 1

                if len(memory) >= max(BATCH_SIZE, WARMUP_STEPS):
                    states, actions, rewards, next_states, dones = (
                        memory.sample(BATCH_SIZE)
                    )

                    states = torch.as_tensor(
                        np.array(states), dtype=torch.float32, device=DEVICE
                    )
                    actions = torch.tensor(
                        actions, dtype=torch.long, device=DEVICE
                    )
                    rewards = torch.tensor(
                        rewards, dtype=torch.float32, device=DEVICE
                    )
                    next_states = torch.as_tensor(
                        np.array(next_states),
                        dtype=torch.float32,
                        device=DEVICE
                    )
                    dones = torch.tensor(
                        dones, dtype=torch.float32, device=DEVICE
                    )

                    current_q = policy_net(states).gather(
                        1, actions.unsqueeze(1)
                    ).squeeze(1)

                    with torch.no_grad():
                        next_q = target_net(next_states).max(dim=1)[0]
                        target_q = rewards + GAMMA * next_q * (1 - dones)

                    loss = F.smooth_l1_loss(current_q, target_q)

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        policy_net.parameters(), 10.0
                    )
                    optimizer.step()

                if total_steps % TARGET_UPDATE == 0:
                    target_net.load_state_dict(
                        policy_net.state_dict()
                    )

            print(
                f"Episode {episode}/{EPISODES} | "
                f"Reward: {episode_reward:.1f} | "
                f"Epsilon: {epsilon:.3f} | "
                f"Steps: {total_steps}"
            )

            if episode_reward > best_reward:
                best_reward = episode_reward
                torch.save(
                    policy_net.state_dict(),
                    "checkpoints/best_mario_dqn.pth"
                )

            if episode % 25 == 0:
                torch.save(
                    policy_net.state_dict(),
                    "checkpoints/mario_dqn_latest.pth"
                )

    finally:
        env.close()

    print("Training complete!")
    print("Best episode reward:", best_reward)


if __name__ == "__main__":
    train()
