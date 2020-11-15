from .WrappedRadioWindow import WrappedRadioWindow
from .WrappedDownloadWindow import WrappedDownloadWindow
from .WrappedFinishWindow import WrappedFinishWindow

#from..StartingApp.WrappedStartingWindow import WrappedStartingWindow

# logging decorator
import sys
sys.path.append("...")
from logs.logger import debug


class BioequivalenceApp():

    def __init__(self, menu_window):
        self.settings = {}
        self.menu_window = menu_window
        self.radio_window = WrappedRadioWindow()
        self.down_window = WrappedDownloadWindow()
        self.finish_window = WrappedFinishWindow()

        self.__build_connections(
            [self.menu_window, self.radio_window, self.down_window, self.finish_window])

    def __build_connections(self, ordered_windows):

        ordered_windows[0].child = ordered_windows[1]
        ordered_windows[0].parent = ordered_windows[-1]

        ordered_windows[-1].child = ordered_windows[0]
        ordered_windows[-1].parent = ordered_windows[-2]

        for i in range(1, len(ordered_windows) - 1):
            ordered_windows[i].child = ordered_windows[i + 1]
            ordered_windows[i].parent = ordered_windows[i - 1]

    @debug
    def start(self):
        self.radio_window.show()
