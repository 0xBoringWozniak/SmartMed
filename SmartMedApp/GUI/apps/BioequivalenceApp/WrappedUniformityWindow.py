import pickle
import threading

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (
    QWidget, QToolTip, QPushButton, QApplication, QMessageBox)

from .UniformityWindow import UniformityWindow


class WrappedUniformityWindow(UniformityWindow, QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.__build_buttons()
        self.radioButtonF.setChecked(True)
        self.setWindowTitle(' ')

    def __build_buttons(self):
        self.pushButtonDone.clicked.connect(self.done)
        self.pushButtonBack.clicked.connect(self.back)
        # self.radioButton_cross.clicked.connect(self.cross)
        # self.radioButton_parall.clicked.connect(self.parall)

    def back(self):
        self.hide()
        self.parent_parral.show()

    def done(self):
        with open('settings.py', 'rb') as f:
            settings = pickle.load(f)
        if self.radioButtonF.isChecked():
            settings['MODULE_SETTINGS']['uniformity'] = 'F'
        else:
            settings['MODULE_SETTINGS']['uniformity'] = 'Leven'
        with open('settings.py', 'wb') as f:
            pickle.dump(settings, f)
        self.hide()
        self.child_parral.show()
