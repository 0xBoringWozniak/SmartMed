import pickle

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (
    QWidget, QToolTip, QPushButton, QApplication, QMessageBox, )

from .RadioWindow import RadioWindow


class WrappedRadioWindow(RadioWindow, QtWidgets.QMainWindow):
   
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.__build_buttons()
        self.setWindowTitle('Что-то там')
        self.settings = { 'MODULE': 'PREDICT'}
       
       

    def __build_buttons(self):
        self.pushButtonNext.clicked.connect(self.next)
        self.pushButtonBack.clicked.connect(self.back)

    def back(self):
        self.hide()
        self.parent.show()

    def next(self):
        with open('settings.py', 'wb') as f:
            pickle.dump(self.settings, f)
        self.hide()
        self.child.show()
