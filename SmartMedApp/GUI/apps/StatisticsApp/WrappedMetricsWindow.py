import pickle

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import (QWidget, QToolTip, QPushButton, QApplication, QMessageBox)

from .MetricsWindow import MetricsWindow


class WrappedMetricsWindow(MetricsWindow, QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle('Метрики')
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

        self.checkBoxCount.setChecked(True)
        self.checkBoxMean.setChecked(True)
        self.checkBoxStd.setChecked(True)
        self.checkBoxMax.setChecked(True)
        self.checkBoxMin.setChecked(True)
        self.checkBoxQuants.setChecked(True)

        self.__build_buttons()

    def __build_buttons(self):
        self.pushButtonBack.clicked.connect(self.back)
        self.pushButtonNext.clicked.connect(self.next)
        self.checkBoxCount.clicked.connect(self.check_count)
        self.checkBoxMean.clicked.connect(self.check_mean)
        self.checkBoxStd.clicked.connect(self.check_std)
        self.checkBoxMax.clicked.connect(self.check_max)
        self.checkBoxMin.clicked.connect(self.check_min)
        self.checkBoxQuants.clicked.connect(self.check_quants)


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
        if self.checkBoxCount.isChecked():
            self.checkBoxCount.setChecked(True)
            self.settings['count'] = True
        else:
            self.checkBoxCount.setChecked(False)
            self.settings['count'] = False

    def check_mean(self):
        if self.checkBoxMean.isChecked():
            self.checkBoxMean.setChecked(True)
            self.settings['mean'] = True
        else:
            self.checkBoxMean.setChecked(False)
            self.settings['mean'] = False

    def check_std(self):
        if self.checkBoxStd.isChecked():
            self.checkBoxStd.setChecked(True)
            self.settings['std'] = True
        else:
            self.checkBoxStd.setChecked(False)
            self.settings['std'] = False

    def check_max(self):
        if self.checkBoxMax.isChecked():
            self.checkBoxMax.setChecked(True)
            self.settings['max'] = True
        else:
            self.checkBoxMax.setChecked(False)
            self.settings['max'] = False

    def check_min(self):
        if self.checkBoxMin.isChecked():
            self.checkBoxMin.setChecked(True)
            self.settings['min'] = True
        else:
            self.checkBoxMin.setChecked(False)
            self.settings['min'] = False

    def check_quants(self):
        if self.checkBoxQuants.isChecked():
            self.checkBoxQuants.setChecked(True)
            self.settings['25%'] = True
            self.settings['50%'] = True
            self.settings['75%'] = True
        else:
            self.checkBoxQuants.setChecked(False)
            self.settings['25%'] = False
            self.settings['50%'] = False
            self.settings['75%'] = False
