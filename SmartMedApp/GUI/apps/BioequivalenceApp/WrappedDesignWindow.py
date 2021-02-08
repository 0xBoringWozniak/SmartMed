import pickle

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (
    QWidget, QToolTip, QPushButton, QApplication, QMessageBox)

from .DesignWindow import DesignWindow



class WrappedDesignWindow(DesignWindow, QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.__build_buttons()
        self.setWindowTitle('Что-то там')

    def __build_buttons(self):
        self.pushButtonNext.clicked.connect(self.next)
        self.pushButtonBack.clicked.connect(self.back)
        #self.radioButton_cross.clicked.connect(self.cross)
        #self.radioButton_parall.clicked.connect(self.parall)

    def back(self):
        self.hide()
        self.parent.show()

    def next(self):
        with open('settings.py', 'rb') as f:
            settings = pickle.load(f)
        if self.radioButton_parall.isChecked():
            settings['MODULE_SETTINGS']['design'] = 'parallel'
        else:
            settings['MODULE_SETTINGS']['design'] = 'cross'
        with open('settings.py', 'wb') as f:
            pickle.dump(settings, f)
        self.hide()
        self.child.show()


