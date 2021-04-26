from abc import ABCMeta, abstractmethod

class Attack(metaclass=ABCMeta):
    def __init__(self, model):
        self.model = model

    @abstractmethod
    def perturb(self, x, y):
        pass
