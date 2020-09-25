from PyQt5 import QtWidgets
from PyQt5.QtWidgets import (QWidget, QToolTip, QPushButton, QApplication)
from .MainStatsWindow import MainStatsWindow


class MainStatsWindowLogic(MainStatsWindow, QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.__build_buttons()

    def __build_buttons(self):
        self.pushButton.clicked.connect(self.next)

    def next(self):
        self.hide()
        self.child.show()