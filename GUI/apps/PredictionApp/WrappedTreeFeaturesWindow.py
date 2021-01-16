import pickle
import threading
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (
    QWidget, QToolTip, QPushButton, QApplication, QMessageBox, )

from .TreeFeaturesWindow import TreeFeaturesWindow


class WrappedTreeFeaturesWindow(TreeFeaturesWindow, QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.__build_buttons()

    def __build_buttons(self):
        self.pushButtonNext.clicked.connect(self.next)
        self.pushButtonBack.clicked.connect(self.back)

    def back(self):
        self.hide()
        self.parent.show()

    def next(self):
        self.hide()
        self.child.show()


        