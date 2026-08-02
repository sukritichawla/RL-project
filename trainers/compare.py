class Comparison:

    def __init__(self):

        self.results = {}

    def add_result(self, name, score):

        self.results[name] = score

    def show(self):

        print("\nAlgorithm Comparison\n")

        ranking = sorted(

            self.results.items(),

            key=lambda x: x[1],

            reverse=True

        )

        for algorithm, score in ranking:

            print(

                f"{algorithm:<20} {score:.2f}"

            )