import matplotlib.pyplot as plt


def plot_rewards(rewards):

    plt.figure(figsize=(8,5))

    plt.plot(rewards)

    plt.xlabel("Episode")

    plt.ylabel("Reward")

    plt.title("Training Reward")

    plt.grid()

    plt.show()