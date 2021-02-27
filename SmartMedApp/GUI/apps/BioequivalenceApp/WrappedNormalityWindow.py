import pickle

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (
    QWidget, QToolTip, QPushButton, QApplication, QMessageBox)

from .NormalityWindow import NormalityWindow



class WrappedNormalityWindow(NormalityWindow, QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.__build_buttons()
        self.radioButtonColm.setChecked(True)
        #self.setWindowTitle('Что-то там')
    

    def __build_buttons(self):
        self.pushButton.clicked.connect(self.next)
        self.pushButton_2.clicked.connect(self.back)
        #self.radioButton_cross.clicked.connect(self.cross)
        #self.radioButton_parall.clicked.connect(self.parall)

    def back(self):
        self.hide()
        with open('settings.py', 'rb') as f:
            design = pickle.load(f)
        if design['MODULE_SETTINGS']['design'] == 'parallel':
            self.parent_parral.show()
        else:
            self.parent_cross.show()

    def next(self):
        with open('settings.py', 'rb') as f:
            settings = pickle.load(f)
        if self.radioButtonColm.isChecked():
            settings['MODULE_SETTINGS']['normality'] = 'Kolmogorov'
        else:
            settings['MODULE_SETTINGS']['normality'] = 'Shapiro'
        with open('settings.py', 'wb') as f:
            pickle.dump(settings, f)
        self.hide()
        self.child_parral.show()


