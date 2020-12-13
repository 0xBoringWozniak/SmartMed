from .WrappedRadioWindow import WrappedRadioWindow
from .WrappedDownloadWindow import WrappedDownloadWindow
from .WrappedChoiceWindow import WrappedChoiceWindow
from .WrappedValueWindow import WrappedValueWindow
from .WrappedRockValueWindow import WrappedRockValueWindow
from .WrappedLinearGraphWindow import WrappedLinearGraphWindow

#from..StartingApp.WrappedStartingWindow import WrappedStartingWindow

# logging decorator
import sys
sys.path.append("...")
from logs.logger import debug


class PredictionApp():

    def __init__(self, menu_window):
        self.settings = {}
        self.menu_window = menu_window
        self.radio_window = WrappedRadioWindow()
        self.down_window = WrappedDownloadWindow()
        self.choice_window = WrappedChoiceWindow()
        self.regression_value_window = WrappedValueWindow()
        self.rock_value_window = WrappedRockValueWindow()
        self.linear_graph_window = WrappedLinearGraphWindow()

        self.__build_connections(
            [self.menu_window, self.down_window, self.radio_window, self.choice_window])

    def __build_connections(self, ordered_windows):

        ordered_windows[0].child = ordered_windows[1]
        ordered_windows[0].parent = ordered_windows[-1]

        ordered_windows[-1].child = ordered_windows[0]
        ordered_windows[-1].parent = ordered_windows[-2]

        for i in range(1, len(ordered_windows) - 1):
            ordered_windows[i].child = ordered_windows[i + 1]
            ordered_windows[i].parent = ordered_windows[i - 1]
        self.choice_window.child_regression = self.regression_value_window
        self.regression_value_window.parent_regression = self.choice_window
        self.choice_window.child_rock = self.rock_value_window
        self.rock_value_window.parent_rock = self.choice_window
        self.regression_value_window.child_linear = self.linear_graph_window
        #self.linear_graph_window.parent_linear = self.
    @debug
    def start(self):
        self.down_window.show()
