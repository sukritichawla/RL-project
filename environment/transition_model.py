class TransitionModel:

    def build(self):

        return {

            "num_states": 3,

            "num_actions": 2,

            "transitions": {

                0: {

                    0: [(1.0, 1, 20, False)],

                    1: [(1.0, 0, -10, False)]

                },

                1: {

                    0: [(1.0, 2, 30, False)],

                    1: [(1.0, 0, -5, False)]

                },

                2: {

                    0: [(1.0, 2, 20, True)],

                    1: [(1.0, 1, 5, False)]

                }

            }

        }