class ConvergenceMetrics:

    @staticmethod
    def convergence_episode(rewards):

        best = max(rewards)

        for i, r in enumerate(rewards):

            if r >= 0.95 * best:

                return i

        return len(rewards)