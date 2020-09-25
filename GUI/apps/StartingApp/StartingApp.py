import os

from .StartingWindowLogic import StartingWindowLogic

# logging decorator
import sys
sys.path.append("...")
from logs.logger import debug


class StartingApp():
    def __init__(self):
        self.settings = {}
        self.startingWindow = StartingWindowLogic()

    def start(self):
        self.startingWindow.show()
        return self.startingWindow.settings
