from .WrappedStartingWindow import WrappedStartingWindow

# logging decorator
from SmartMedApp.logs.logger import debug


class StartingApp():

    @debug
    def __init__(self):
        self.startingWindow = WrappedStartingWindow()

    @debug
    def start(self):
        self.startingWindow.show()
