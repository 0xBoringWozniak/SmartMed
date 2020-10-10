from .WrappedStartingWindow import WrappedStartingWindow

# logging decorator
import sys
sys.path.append("...")
from logs.logger import debug


class StartingApp():

    def __init__(self):

        self.startingWindow = WrappedStartingWindow()

    @debug
    def start(self):
        self.startingWindow.show()
        return self.startingWindow.settings
