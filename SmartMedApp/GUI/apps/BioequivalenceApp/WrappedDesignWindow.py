import pickle

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (
    QWidget, QToolTip, QPushButton, QApplication, QMessageBox)

from .DesignWindow import DesignWindow
from ..utils import remove_if_exists


class WrappedDesignWindow(DesignWindow, QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.__build_buttons()
        self.setWindowTitle('Выбор плана')
        self.radioButton_cross.setChecked(True)
        self.settings = {'MODULE': 'BIOEQ', 'MODULE_SETTINGS': {
            'path_test': '', 'path_ref': '', 'design': ''}}

    def __build_buttons(self):
        self.pushButtonNext.clicked.connect(self.next)
        self.pushButtonBack.clicked.connect(self.back)

    def back(self):
        self.hide()
        remove_if_exists()
        self.parent.show()

    def next(self):
        if self.radioButton_parall.isChecked():
            self.settings['MODULE_SETTINGS']['design'] = 'parallel'
            self.child_parral.show()
        else:
            self.settings['MODULE_SETTINGS']['design'] = 'cross'
            self.child_cross.show()
        with open('settings.py', 'wb') as f:
            pickle.dump(self.settings, f)
        self.hide()
