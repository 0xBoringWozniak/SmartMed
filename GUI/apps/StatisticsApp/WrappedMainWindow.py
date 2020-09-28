from PyQt5 import QtWidgets
from PyQt5.QtWidgets import (QWidget, QToolTip, QPushButton, QApplication)
from .MainWindow import MainWindow


class WrappedMainWindow(MainWindow, QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.__build_buttons()

    def __build_buttons(self):
        self.pushButtonNext.clicked.connect(self.next)
        self.checkBoxChoice1.clicked.connect(self.choice1)
        self.checkBoxChoice2.clicked.connect(self.choice2)

    def next(self):
        self.hide()
        self.child.show()

    def choice1(self):
    	print('1')

    def choice2(self):
    	print('2')
