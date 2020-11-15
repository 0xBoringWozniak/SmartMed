import pickle

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (
    QWidget, QToolTip, QPushButton, QApplication, QMessageBox, )

from .DownloadWindow import DownloadWindow


class WrappedDownloadWindow(DownloadWindow, QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.__build_buttons()
        self.setWindowTitle('Что-то там')
       

    def __build_buttons(self):
        #плохо с неймингом, надо переделать бек некст
        self.pushButtonBack.clicked.connect(self.next)
        self.pushButtonNext.clicked.connect(self.back)

    def back(self):
        self.hide()
        self.parent.show()

    def next(self):
        
        self.hide()
        self.child.show()
