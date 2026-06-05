from abc import ABC, abstractmethod

class BaseGene(ABC):
    def __str__(self): return f'base'
    def __repr__(self): return str(self)
    def __init__(self): pass

    @abstractmethod
    def executor(self, executable): pass