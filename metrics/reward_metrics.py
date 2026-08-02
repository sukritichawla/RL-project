class RewardMetrics:

    @staticmethod
    def cumulative_reward(rewards):

        return sum(rewards)

    @staticmethod
    def average_reward(rewards):

        return sum(rewards) / len(rewards)