import os

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import (QWidget, QToolTip, QPushButton, QApplication)

from .predict1 import *

# logging decorator
import sys
sys.path.append("...")
from logs.logger import debug

class PredictionApp(Ui_MainWindow_predict, QtWidgets.QMainWindow):
    switch_window = QtCore.pyqtSignal()
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.__build_buttons()

    def __build_buttons(self):
        self.pushButton.clicked.connect(self.pushbutton_handler)


    def pushbutton_handler(self):
        self.switch_window.emit()