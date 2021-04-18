import pickle

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import (
    QWidget, QToolTip, QPushButton, QApplication, QMessageBox)

from .StartingWindow import *
from ..StatisticsApp.StatisticsAppController import StatisticsApp
from ..BioequivalenceApp.BioequivalenceAppController import BioequivalenceApp
from ..PredictionApp.PredictionAppController import PredictionApp


class WrappedStartingWindow(StartingWindow, QtWidgets.QMainWindow):

    def __init__(self):

        #self.settings = {}
        super().__init__()
        self.setupUi(self)
        self.__build_buttons()
        self.setWindowTitle('SmartMed')

    def __build_buttons(self):
        # create button and add signals
        self.pushButtonStat.clicked.connect(self.button_stats)
        self.pushButtonPred.clicked.connect(self.button_prediction)
        self.pushButtonBioeq.clicked.connect(self.button_bioeq)
        self.pushButtonDone.clicked.connect(self.done)

    def done(self):
        self.close()

    def button_stats(self):
        self.hide()
        app = StatisticsApp(menu_window=self)
        app.start()

        # update settings
        return app.settings

    def button_prediction(self):
        self.hide()
        app = PredictionApp(menu_window=self)
        app.start()

    def button_bioeq(self):
        self.hide()
        app = BioequivalenceApp(menu_window=self)
        app.start()
