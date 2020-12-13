import pickle
import threading
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (
    QWidget, QToolTip, QPushButton, QApplication, QMessageBox, )

from .ChoiceWindow import ChoiceWindow


class WrappedChoiceWindow(ChoiceWindow, QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.__build_buttons()
        self.setWindowTitle('Выбор регрессии')
        #self.radioButtonLinear.SetChecked(True)

    def __build_buttons(self):
        self.pushButtonNext.clicked.connect(self.next)
        self.pushButtonBack.clicked.connect(self.back)

    def back(self):
        self.hide()
        self.parent.show()

    def next(self):
        self.hide()
        if self.radioButtonLinear.isChecked():
            self.child_regression.show()
        elif self.radioButtonLogit.isChecked():
            self.child_regression.show()
        elif self.radioButtonPol.isChecked():
            self.child_regression.show()
        else:
            self.child_rock.show()

        
        
