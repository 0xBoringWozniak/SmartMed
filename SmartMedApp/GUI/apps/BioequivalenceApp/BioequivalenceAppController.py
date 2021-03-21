from .WrappedDesignWindow import WrappedDesignWindow
from .WrappedDownloadWindow import WrappedDownloadWindow
from .WrappedFinishWindow import WrappedFinishWindow
from .WrappedTablesWindow import WrappedTablesWindow
from .WrappedGraphsWindow import WrappedGraphsWindow
from .WrappedUniformityWindow import WrappedUniformityWindow
from .WrappedNormalityWindow import WrappedNormalityWindow
from .WrappedDownloadWindowCross import WrappedDownloadWindowCross
from .WrappedTablesWindowCross import WrappedTablesWindowCross
from .WrappedGraphsWindowCross import WrappedGraphsWindowCross
#from..StartingApp.WrappedStartingWindow import WrappedStartingWindow

# logging decorator
from SmartMedApp.logs.logger import debug


class BioequivalenceApp():

    @debug
    def __init__(self, menu_window):
        self.settings = {}
        self.menu_window = menu_window
        self.design_window = WrappedDesignWindow()
        self.down_window_parral = WrappedDownloadWindow()
        self.tables_window_parral = WrappedTablesWindow()
        self.graphs_window_parral = WrappedGraphsWindow()
        self.normality_window_parral = WrappedNormalityWindow()
        self.uniformity_window_parral = WrappedUniformityWindow()

        self.down_window_cross = WrappedDownloadWindowCross()
        self.tables_window_cross = WrappedTablesWindowCross()
        self.graphs_window_cross = WrappedGraphsWindowCross()
        self.normality_window_cross = WrappedNormalityWindow()

        self.__build_connections()

    @debug
    def __build_connections(self):
        '''
        ordered_windows[0].child = ordered_windows[1]
        ordered_windows[0].parent = ordered_windows[-1]

        ordered_windows[-1].child = ordered_windows[0]
        ordered_windows[-1].parent = ordered_windows[-2]

        for i in range(1, len(ordered_windows) - 1):
            ordered_windows[i].child = ordered_windows[i + 1]
            ordered_windows[i].parent = ordered_windows[i - 1]

         [self.menu_window, self.design_window, self.down_window,  self.tables_window,
                                        self.graphs_window, self.normality_window, self.uniformity_window]


        '''
        self.menu_window.child = self.design_window
        self.design_window.parent = self.menu_window
        self.design_window.child_parral = self.down_window_parral
        self.down_window_parral.parent_parral = self.design_window
        self.down_window_parral.child_parral = self.normality_window_parral
        self.normality_window_parral.parent_parral = self.down_window_parral
        self.normality_window_parral.child_parral = self.uniformity_window_parral
        self.uniformity_window_parral.parent_parral = self.normality_window_parral
        self.uniformity_window_parral.child_parral = self.tables_window_parral
        self.tables_window_parral.parent_parral = self.uniformity_window_parral
        self.tables_window_parral.child_parral = self.graphs_window_parral
        self.graphs_window_parral.parent_parral = self.tables_window_parral
        self.graphs_window_parral.child_parral = self.menu_window

        self.design_window.child_cross = self.down_window_cross
        self.down_window_cross.parent_cross = self.design_window
        self.down_window_cross.child_cross = self.normality_window_cross
        self.normality_window_cross.parent_parral = self.down_window_cross
        self.normality_window_cross.child_parral = self.tables_window_cross
        self.tables_window_cross.parent_cross = self.normality_window_cross
        self.tables_window_cross.child_cross = self.graphs_window_cross
        self.graphs_window_cross.parent_cross = self.tables_window_cross
        self.graphs_window_cross.child_cross = self.menu_window

    @debug
    def start(self):
        self.design_window.show()
