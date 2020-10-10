import pickle

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import (QWidget, QToolTip, QPushButton, QApplication)

from .SecondWindow import SecondWindow


class WrappedSecondWindow(SecondWindow, QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.settings = {'metrics':
                         {
                             'count': True,
                             'mean': True,
                             'std': True,
                             'max': True,
                             'min': True,
                             '25%': True,
                             '50%': True,
                             '75%': True
                         }
                         }

        self.checkBox_2.setChecked(True)
        self.checkBox_3.setChecked(True)
        self.checkBox_4.setChecked(True)
        self.checkBox_5.setChecked(True)
        self.checkBox_6.setChecked(True)
        self.checkBox_7.setChecked(True)

        self.__build_buttons()

    def __build_buttons(self):
        self.pushButtonBack.clicked.connect(self.back)
        self.pushButtonNext.clicked.connect(self.next)
        self.checkBox_2.clicked.connect(self.check_count)
        self.checkBox_3.clicked.connect(self.check_mean)
        self.checkBox_4.clicked.connect(self.check_std)
        self.checkBox_5.clicked.connect(self.check_max)
        self.checkBox_6.clicked.connect(self.check_min)
        self.checkBox_7.clicked.connect(self.check_quants)

    def back(self):
        self.hide()
        self.leaf_1.show()

    def next(self):
        with open('settings.py', 'rb') as f:
            data = pickle.load(f)
            data['MODULE_SETTINGS'].update(self.settings)

        with open('settings.py', 'wb') as f:
            pickle.dump(data, f)

        self.hide()
        self.leaf_3.show()

    def check_count(self):
        if self.checkBox_2.isChecked():
            self.checkBox_2.setChecked(True)
            self.settings['metrics']['count'] = True
        else:
            self.checkBox_2.setChecked(False)
            self.settings['metrics']['count'] = False

    def check_mean(self):
        if self.checkBox_3.isChecked():
            self.checkBox_3.setChecked(True)
            self.settings['metrics']['mean'] = True
        else:
            self.checkBox_3.setChecked(False)
            self.settings['metrics']['mean'] = False

    def check_std(self):
        if self.checkBox_4.isChecked():
            self.checkBox_4.setChecked(True)
            self.settings['metrics']['std'] = True
        else:
            self.checkBox_4.setChecked(False)
            self.settings['metrics']['std'] = False

    def check_max(self):
        if self.checkBox_5.isChecked():
            self.checkBox_5.setChecked(True)
            self.settings['metrics']['max'] = True
        else:
            self.checkBox_5.setChecked(False)
            self.settings['metrics']['max'] = False

    def check_min(self):
        if self.checkBox_6.isChecked():
            self.checkBox_6.setChecked(True)
            self.settings['metrics']['min'] = True
        else:
            self.checkBox_6.setChecked(False)
            self.settings['metrics']['min'] = False

    def check_quants(self):
        if self.checkBox_7.isChecked():
            self.checkBox_7.setChecked(True)
            self.settings['metrics']['25%'] = True
            self.settings['metrics']['50%'] = True
            self.settings['metrics']['75%'] = True
        else:
            self.checkBox_7.setChecked(False)
            self.settings['metrics']['25%'] = False
            self.settings['metrics']['50%'] = False
            self.settings['metrics']['75%'] = False
