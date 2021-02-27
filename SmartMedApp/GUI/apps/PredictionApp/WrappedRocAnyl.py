import pickle
import threading
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (
    QWidget, QToolTip, QPushButton, QApplication, QMessageBox, )

from .RocAnyl import RocAnyl


class WrappedRocAnyl(RocAnyl, QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.checkBoxAccur.setChecked(True)
        self.checkBoxConf.setChecked(True)
        self.checkBoxF.setChecked(True)
        self.checkBoxPrecision.setChecked(True)
        self.checkBoxRecall.setChecked(True)
        self.checkBoxTrashhold.setChecked(True)
        self.settings = {'accuracy': True,
                            'confidence': True,
                            'F':True,
                            'precision':True,
                            'recall': True,
                            'trashhold':True}
        self.__build_buttons()

    def __build_buttons(self):
        self.pushButtonDone.clicked.connect(self.next)
        self.pushButtonBack.clicked.connect(self.back)

    def back(self):
        self.hide()
        self.parent.show()

    def next(self):
        if self.checkBoxAccur.isChecked() != True:
            self.settings['accuracy'] = False
        if self.checkBoxConf.isChecked() != True:
            self.settings['confidence'] = False
        if self.checkBoxF.isChecked() != True:
            self.settings['F'] = False
        if self.checkBoxPrecision.isChecked() != True:
            self.settings['precision'] = False
        if self.checkBoxRecall.isChecked() != True:
            self.settings['recall'] = False
        if self.checkBoxTrashhold.isChecked() != True:
            self.settings['trashhold'] = False
        with open('settings.py', 'rb') as f:
            data = pickle.load(f)
            data['MODULE_SETTINGS'].update(self.settings)
        with open('settings.py', 'wb') as f:
            pickle.dump(data, f)
        self.hide()
        self.child.show()
        
