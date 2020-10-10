import pickle

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import (QWidget, QToolTip, QPushButton, QApplication)

from .ThirdWindow import ThirdWindow


class WrappedThirdWindow(ThirdWindow, QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.settings = {'graphs': {
            'linear': True,
            'log': True,
            'corr': True,
            'heatmap': True,
            'scatter': True,
            'hist': True,
            'box': True
        }
        }
        self.checkBox_2.setChecked(True)
        self.checkBox_3.setChecked(True)
        self.checkBox_4.setChecked(True)
        self.checkBox_5.setChecked(True)
        self.checkBox_6.setChecked(True)
        self.checkBox_7.setChecked(True)
        self.checkBox_8.setChecked(True)
        self.__build_buttons()

    def __build_buttons(self):
        self.pushButton.clicked.connect(self.back)
        self.pushButtonDone.clicked.connect(self.done)
        self.checkBox_2.clicked.connect(self.check_linear)
        self.checkBox_4.clicked.connect(self.check_log)
        self.checkBox_6.clicked.connect(self.check_corr)
        self.checkBox_3.clicked.connect(self.check_heatmap)
        self.checkBox_8.clicked.connect(self.check_scatter)
        self.checkBox_5.clicked.connect(self.check_hist)
        self.checkBox_7.clicked.connect(self.check_box)

    def back(self):
        self.hide()
        self.leaf_4.show()

    def done(self):
        with open('settings.py', 'rb') as f:
            data = pickle.load(f)
            data['MODULE_SETTINGS'].update(self.settings)

        with open('settings.py', 'wb') as f:
            pickle.dump(data, f)

        self.close()

    def check_linear(self):
        if self.checkBox_2.isChecked():
            self.checkBox_2.setChecked(True)
            self.settings['graphs']['linear'] = True
        else:
            self.checkBox_2.setChecked(False)
            self.settings['graphs']['linear'] = False

    def check_log(self):
        if self.checkBox_4.isChecked():
            self.checkBox_4.setChecked(True)
            self.settings['graphs']['log'] = True
        else:
            self.checkBox_4.setChecked(False)
            self.settings['graphs']['log'] = False

    def check_corr(self):
        if self.checkBox_6.isChecked():
            self.checkBox_6.setChecked(True)
            self.settings['graphs']['corr'] = True
        else:
            self.checkBox_6.setChecked(False)
            self.settings['graphs']['corr'] = False

    def check_heatmap(self):
        if self.checkBox_3.isChecked():
            self.checkBox_3.setChecked(True)
            self.settings['graphs']['heatmap'] = True
        else:
            self.checkBox_3.setChecked(False)
            self.settings['graphs']['heatmap'] = False

    def check_scatter(self):
        if self.checkBox_8.isChecked():
            self.checkBox_8.setChecked(True)
            self.settings['graphs']['scatter'] = True
        else:
            self.checkBox_8.setChecked(False)
            self.settings['graphs']['scatter'] = False

    def check_hist(self):
        if self.checkBox_5.isChecked():
            self.checkBox_5.setChecked(True)
            self.settings['graphs']['hist'] = True
        else:
            self.checkBox_5.setChecked(False)
            self.settings['graphs']['hist'] = False

    def check_box(self):
        if self.checkBox_7.isChecked():
            self.checkBox_7.setChecked(True)
            self.settings['graphs']['box'] = True
        else:
            self.checkBox_7.setChecked(False)
            self.settings['graphs']['box'] = False
