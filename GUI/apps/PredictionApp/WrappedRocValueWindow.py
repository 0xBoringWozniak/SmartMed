import pickle

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (
    QWidget, QToolTip, QPushButton, QApplication, QMessageBox, QTableWidget)
from .RocValueWindow import RocValueWindow



class WrappedRocValueWindow(RocValueWindow, QtWidgets.QMainWindow):
   
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
        var = self.comboBox.currentText()
        with open('settings.py', 'rb') as f:
                data = pickle.load(f)
        data['MODULE_SETTINGS'].update({'variable': var})
        with open('settings.py', 'wb') as f:
            pickle.dump(data, f)
        self.hide()
        self.child.show()


