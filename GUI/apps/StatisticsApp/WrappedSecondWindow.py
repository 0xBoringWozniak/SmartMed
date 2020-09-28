from PyQt5 import QtWidgets
from PyQt5.QtWidgets import (QWidget, QToolTip, QPushButton, QApplication)
from .SecondWindow import SecondWindow


class WrappedSecondWindow(SecondWindow, QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        
        self.__build_buttons()

    def __build_buttons(self):
        self.pushButton.clicked.connect(self.back)

    def back(self):
        self.hide()
        self.parent.show()
