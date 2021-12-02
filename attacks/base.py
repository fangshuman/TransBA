from abc import ABCMeta, abstractmethod

class Attack(metaclass=ABCMeta):
    config = {}
    def __init__(self, model):
        self.model = model

    def register_hook(self):
        pass

    @abstractmethod
    def perturb(self, x, y):
        pass
