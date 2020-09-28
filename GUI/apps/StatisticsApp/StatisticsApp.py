from .WrappedMainWindow import WrappedMainWindow
from .WrappedSecondWindow import WrappedSecondWindow

# logging decorator
import sys
sys.path.append("...")
from logs.logger import debug


class StatisticsApp():
    def __init__(self):
        self.settings = {}

        self.main_window = WrappedMainWindow()
        self.second_window = WrappedSecondWindow()

        self.__build_connections()

    def __build_connections(self):    
        self.main_window.child = self.second_window
        self.second_window.parent = self.main_window

    def start(self):
        self.main_window.show()

