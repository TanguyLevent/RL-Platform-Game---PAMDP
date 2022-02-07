from abc import ABC, abstractmethod

class Agent(ABC):
    """
    Defines a basic reinforcement learning agent for OpenAI Gym environments
    """

    def __init__(self, name):
        super().__init__()

        self.name = name

    @abstractmethod
    def create_algorithm(self):
        """Create algorithm."""
        pass

    @abstractmethod
    def test(self):
        """Test algorithm."""
        pass

    @abstractmethod
    def train(self):
        """Train algorithm."""
        pass
