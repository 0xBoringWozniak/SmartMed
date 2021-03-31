import pickle

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (
    QWidget, QToolTip, QPushButton, QApplication, QMessageBox, QTableWidget)
from .RocCurvesWindow import RocCurvesWindow



class WrappedRocCurvesWindow(RocCurvesWindow, QtWidgets.QMainWindow):
   
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.checkBoxAuc.setChecked(True)
        self.checkBoxDiff.setChecked(True)
        self.checkBoxPaint.setChecked(True)
        self.settings = {'auc': True,
                        'diff_graphics': True,
                        'paint': True}
        self.__build_buttons()
        self.setWindowTitle(' ')

    def __build_buttons(self):
        self.pushButtonNext.clicked.connect(self.next)
        self.pushButtonBack.clicked.connect(self.back)

    def back(self):
        self.hide()
        self.parent.show()

    def next(self):
        if self.checkBoxAuc.isChecked() != True:
            self.settings['auc'] = False
        if self.checkBoxDiff.isChecked() != True:
            self.settings['diff_graphics'] = False
        if self.checkBoxPaint.isChecked() != True:
            self.settings['paint'] = False
        with open('settings.py', 'rb') as f:
            data = pickle.load(f)
            data['MODULE_SETTINGS'].update(self.settings)
        with open('settings.py', 'wb') as f:
            pickle.dump(data, f)
        self.hide()
        self.child.show()

