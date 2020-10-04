from .WrappedMainWindow import WrappedMainWindow
from .WrappedSecondWindow import WrappedSecondWindow
from .WrappedThirdWindow import WrappedThirdWindow

# logging decorator
import sys
sys.path.append("...")
from logs.logger import debug


class StatisticsApp():

    def __init__(self):
        self.settings = {}
        self.main_window = WrappedMainWindow()
        self.second_window = WrappedSecondWindow()
        self.third_window = WrappedThirdWindow()

        self.__build_connections()

    def __build_connections(self):
        # list structure
        self.main_window.leaf_2 = self.second_window
        self.second_window.leaf_1 = self.main_window
        self.second_window.leaf_3 = self.third_window
        self.third_window.leaf_4 = self.second_window

    @debug
    def start(self):
        self.main_window.show()
