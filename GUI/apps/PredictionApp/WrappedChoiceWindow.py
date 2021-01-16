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

    def __build_buttons(self):
        self.pushButtonNext.clicked.connect(self.next)
        self.pushButtonBack.clicked.connect(self.back)

    def back(self):
        self.hide()
        self.parent.show()

    def next(self):
        self.hide()
        if self.radioButtonLinear.isChecked() or self.radioButtonLogit.isChecked() or self.radioButtonPol.isChecked():
            self.child_linear.show()
            with open('settings.py', 'rb') as f:
                data = pickle.load(f)
            col = data['MODULE_SETTINGS']['columns']
            self.child_linear.comboBox.addItems(col)
        elif self.radioButtonRoc.isChecked():
            self.child_roc.show()
        else:
            self.child_tree.show()


        
        
