from .WrappedDesignWindow import WrappedDesignWindow
from .WrappedDownloadWindow import WrappedDownloadWindow
from .WrappedFinishWindow import WrappedFinishWindow
from .WrappedTablesWindow import WrappedTablesWindow
from .WrappedGraphsWindow import WrappedGraphsWindow
from .WrappedUniformityWindow import WrappedUniformityWindow
from .WrappedNormalityWindow import WrappedNormalityWindow

#from..StartingApp.WrappedStartingWindow import WrappedStartingWindow

# logging decorator
import sys
from SmartMedApp.logs.logger import debug


class BioequivalenceApp():

    def __init__(self, menu_window):
        self.settings = {}
        self.menu_window = menu_window
        self.design_window = WrappedDesignWindow()
        self.down_window = WrappedDownloadWindow()
        self.tables_window = WrappedTablesWindow()
        self.graphs_window = WrappedGraphsWindow()
        self.normality_window = WrappedNormalityWindow()
        self.uniformity_window = WrappedUniformityWindow()

        self.__build_connections(
            [self.menu_window, self.down_window, self.design_window,  self.tables_window,
                                        self.graphs_window, self.normality_window, self.uniformity_window])

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
