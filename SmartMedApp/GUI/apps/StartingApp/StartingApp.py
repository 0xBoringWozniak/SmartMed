from .WrappedStartingWindow import WrappedStartingWindow

# logging decorator
import sys
from SmartMedApp.logs.logger import debug


class StartingApp():

    def __init__(self):

        self.startingWindow = WrappedStartingWindow()

    @debug
    def start(self):
        self.startingWindow.show()
