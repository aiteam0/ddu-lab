from .state import ParseState
from abc import ABC, abstractmethod


class BaseNode(ABC):
    def __init__(self, verbose=False, **kwargs):
        self.name = self.__class__.__name__
        self.verbose = verbose

    @abstractmethod
    def execute(self, state: ParseState) -> ParseState:
        pass

    def log(self, message: str, **kwargs):
        if self.verbose:
            print(f"[{self.name}] {message}")
            for key, value in kwargs.items():
                print(f"  {key}: {value}")

    def __call__(self, state: ParseState) -> ParseState:
        return self.execute(state)
