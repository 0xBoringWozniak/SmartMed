import os

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import (QWidget, QToolTip, QPushButton, QApplication)

from .design import *

# logging decorator
import sys
sys.path.append("...")
from logs.logger import debug


class StartingApp(Ui_MainWindow, QtWidgets.QMainWindow):
    def __init__(self):
        self.settings = {}

        super().__init__()
        self.setupUi(self)

        self.__build_buttons()

    def __build_buttons(self):
        self.pushButton.clicked.connect(self.button_stats)
        self.pushButton.setToolTip('heelp')
        self.pushButton_2.clicked.connect(self.button_prediction)
        self.pushButton_3.clicked.connect(self.button_bioeq)
        self.pushButton_4.clicked.connect(self.button_done)

        self.commandLinkButton.clicked.connect(self.path_to_file)

    def clickMethod(self):
        print('PyQt')

    def button_stats(self):
        self.settings['MODULE'] = 'STATISTICS'

    def button_prediction(self):
        self.settings['MODULE'] = 'PREDICTION'

    def button_bioeq(self):
        self.settings['MODULE'] = 'BIOEQ'

    def path_to_file(self):
        self.settings['data'] = {'preprocessing': {
                                               'AUTO': False,
                                               'fillna': 'mean',
                                               'encoding': 'label_encoding',
                                               'scaling': False
                                              },
                                'path': ''
                                }
        self.settings['data']['path'] = QtWidgets.QFileDialog.getOpenFileName()[0]

    def button_done(self):
        self.close()
