from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """
    Base class for all Reinforcement Learning agents.

    Every RL algorithm must inherit from this class.
    """

    def __init__(self, state_size, action_size):

        self.state_size = state_size
        self.action_size = action_size

        self.name = self.__class__.__name__

    @abstractmethod
    def select_action(self, state):
        """
        Choose an action given the current state.
        """
        pass

    @abstractmethod
    def update(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):
        """
        Update the learning model.
        """
        pass

    def save(self, filepath):
        """
        Save trained model.
        """
        pass

    def load(self, filepath):
        """
        Load trained model.
        """
        pass