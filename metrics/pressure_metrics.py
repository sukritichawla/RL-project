class PressureMetrics:

    @staticmethod
    def average_pressure(values):

        return sum(values) / len(values)

    @staticmethod
    def violations(values):

        count = 0

        for p in values:

            if p < 30 or p > 80:

                count += 1

        return count