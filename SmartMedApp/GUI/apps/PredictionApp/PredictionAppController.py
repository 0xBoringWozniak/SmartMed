from .WrappedRadioWindow import WrappedRadioWindow
from .WrappedDownloadWindow import WrappedDownloadWindow
from .WrappedChoiceWindow import WrappedChoiceWindow
from .WrappedValueWindow import WrappedValueWindow
from .WrappedRocValueWindow import WrappedRocValueWindow
from .WrappedLinearGraphWindow import WrappedLinearGraphWindow
from .WrappedTreeFeaturesWindow import WrappedTreeFeaturesWindow
from .WrappedTreeVisualWindow import WrappedTreeVisualWindow
from.WrappedRocAnyl import WrappedRocAnyl
from .WrappedRocCurvesWindow import WrappedRocCurvesWindow
from .WrappedRocGraphsWindow import WrappedRocGraphsWindow

#from..StartingApp.WrappedStartingWindow import WrappedStartingWindow

# logging decorator
import sys
from SmartMedApp.logs.logger import debug


class PredictionApp():

    def __init__(self, menu_window):
        self.settings = {}
        self.menu_window = menu_window
        self.radio_window = WrappedRadioWindow()
        self.down_window = WrappedDownloadWindow()
        self.choice_window = WrappedChoiceWindow()
        self.value_window = WrappedValueWindow()
        self.roc_value_window = WrappedRocValueWindow()
        self.roc_anyl_window = WrappedRocAnyl()
        self.roc_curves_window = WrappedRocCurvesWindow()
        self.roc_graphs_window = WrappedRocGraphsWindow()
        self.linear_graph_window = WrappedLinearGraphWindow()
        self.tree_features_window = WrappedTreeFeaturesWindow()
        self.tree_visual_window = WrappedTreeVisualWindow()

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

        self.choice_window.child_linear = self.value_window
        self.value_window.parent = self.choice_window
        self.value_window.child_linear = self.linear_graph_window
        self.linear_graph_window.parent = self.value_window
        self.linear_graph_window.child = self.menu_window

        self.choice_window.child_roc = self.roc_value_window
        self.roc_value_window.parent = self.choice_window
        self.roc_value_window.child = self.roc_anyl_window
        self.roc_anyl_window.parent = self.roc_value_window
        self.roc_anyl_window.child = self.roc_curves_window
        self.roc_curves_window.parent = self.roc_anyl_window
        self.roc_curves_window.child = self.roc_graphs_window
        self.roc_graphs_window.parent = self.roc_curves_window
        self.roc_graphs_window.child = self.menu_window

        self.choice_window.child_tree = self.tree_features_window
        self.tree_features_window.parent = self.choice_window
        self.tree_features_window.child = self.tree_visual_window
        self.tree_visual_window.parent = self.tree_features_window
        self.tree_visual_window.child = self.menu_window


        #self.choice_window.child_regression = self.regression_value_window
        #self.regression_value_window.parent_regression = self.choice_window
        #self.choice_window.child_rock = self.rock_value_window
        #self.rock_value_window.parent_rock = self.choice_window
        #self.regression_value_window.child_linear = self.linear_graph_window
        #self.linear_graph_window.parent_linear = self.
    @debug
    def start(self):
        self.down_window.show()
