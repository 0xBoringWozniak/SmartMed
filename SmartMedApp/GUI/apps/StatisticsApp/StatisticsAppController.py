from .WrappedPreprocessingWindow import WrappedPreprocessingWindow
from .WrappedMetricsWindow import WrappedMetricsWindow
from .WrappedVisualizationWindow import WrappedVisualizationWindow
from .WrappedPrepWindow import WrappedRadioWindow
#from..StartingApp.WrappedStartingWindow import WrappedStartingWindow

# logging decorator
from SmartMedApp.logs.logger import debug


class StatisticsApp():

    @debug
    def __init__(self, menu_window):
        self.settings = {}
        self.menu_window = menu_window
        self.down_window = WrappedPreprocessingWindow()
        self.metrics_window = WrappedMetricsWindow()
        self.graphs_window = WrappedVisualizationWindow()
        self.prep_window = WrappedRadioWindow()

        self.__build_connections(
            [self.menu_window, self.down_window, self.prep_window, self.metrics_window, self.graphs_window])

    @debug
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
        self.down_window.show()
