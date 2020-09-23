import os

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import (QWidget, QToolTip, QPushButton, QApplication)

from .predict1 import *

# logging decorator
import sys
sys.path.append("...")
from logs.logger import debug

class PredictionApp(Ui_MainWindow_predict, QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.__build_buttons()

    def __build_buttons(self):
        self.checkBox.clicked.connect(self.button_choice_1)
        self.checkBox_2.clicked.connect(self.button_choice_2)

    
    def button_choice_1(self):
    	print('work1')


    def button_choice_2(self):
    	print('work2')
