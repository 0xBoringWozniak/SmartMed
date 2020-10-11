import pickle

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import (QWidget, QToolTip, QPushButton, QApplication)

from .MetricsWindow import MetricsWindow


class WrappedMetricsWindow(MetricsWindow, QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle('Cтатистические метрики')
        self.settings = {
                             'count': True,
                             'mean': True,
                             'std': True,
                             'max': True,
                             'min': True,
                             '25%': True,
                             '50%': True,
                             '75%': True
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
        self.parent.show()

    def next(self):
        with open('settings.py', 'rb') as f:
            data = pickle.load(f)
            data['MODULE_SETTINGS']['metrics'].update(self.settings)

        with open('settings.py', 'wb') as f:
            pickle.dump(data, f)

        self.hide()
        self.child.show()

    def check_count(self):
        if self.checkBox_2.isChecked():
            self.checkBox_2.setChecked(True)
            self.settings['count'] = True
        else:
            self.checkBox_2.setChecked(False)
            self.settings['count'] = False

    def check_mean(self):
        if self.checkBox_3.isChecked():
            self.checkBox_3.setChecked(True)
            self.settings['mean'] = True
        else:
            self.checkBox_3.setChecked(False)
            self.settings['mean'] = False

    def check_std(self):
        if self.checkBox_4.isChecked():
            self.checkBox_4.setChecked(True)
            self.settings['std'] = True
        else:
            self.checkBox_4.setChecked(False)
            self.settings['std'] = False

    def check_max(self):
        if self.checkBox_5.isChecked():
            self.checkBox_5.setChecked(True)
            self.settings['max'] = True
        else:
            self.checkBox_5.setChecked(False)
            self.settings['max'] = False

    def check_min(self):
        if self.checkBox_6.isChecked():
            self.checkBox_6.setChecked(True)
            self.settings['min'] = True
        else:
            self.checkBox_6.setChecked(False)
            self.settings['min'] = False

    def check_quants(self):
        if self.checkBox_7.isChecked():
            self.checkBox_7.setChecked(True)
            self.settings['25%'] = True
            self.settings['50%'] = True
            self.settings['75%'] = True
        else:
            self.checkBox_7.setChecked(False)
            self.settings['25%'] = False
            self.settings['50%'] = False
            self.settings['75%'] = False
