import pickle

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (
    QWidget, QToolTip, QPushButton, QApplication, QMessageBox, QTableWidget)
from .RockValueWindow import RockValueWindow



class WrappedRockValueWindow(RockValueWindow, QtWidgets.QMainWindow):
   
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.__build_buttons()
       

    def __build_buttons(self):
        self.pushButtonNext.clicked.connect(self.next)
        self.pushButtonBack.clicked.connect(self.back)

    def back(self):
        self.hide()
        self.parent_rock.show()

    def next(self):
        #self.child.show()
        pass

