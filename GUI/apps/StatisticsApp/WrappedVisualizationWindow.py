import pickle
import time

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import (QWidget, QToolTip, QPushButton, QApplication)

from .VisualizationWindow import VisualizationWindow


class WrappedVisualizationWindow(VisualizationWindow, QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle('Визуализация')
        self.settings = {
            'linear': True,
            'log': True,
            'corr': True,
            'heatmap': True,
            'scatter': True,
            'hist': True,
            'box': True
        }
        self.checkBox.setChecked(True)
        self.checkBox_2.setChecked(True)
        self.checkBox_3.setChecked(True)
        self.checkBox_4.setChecked(True)
        self.checkBox_5.setChecked(True)
        self.checkBox_6.setChecked(True)
        self.checkBox_7.setChecked(True)
        self.checkBox_8.setChecked(True)
        self.checkBox_9.setChecked(True)
        self.__build_buttons()

    def __build_buttons(self):
        self.pushButton.clicked.connect(self.back)
        self.pushButtonDone.clicked.connect(self.done)
        self.checkBox.clicked.connect(self.check_bar)
        self.checkBox_2.clicked.connect(self.check_linear)
        self.checkBox_4.clicked.connect(self.check_log)
        self.checkBox_6.clicked.connect(self.check_corr)
        self.checkBox_3.clicked.connect(self.check_heatmap)
        self.checkBox_8.clicked.connect(self.check_scatter)
        self.checkBox_5.clicked.connect(self.check_hist)
        self.checkBox_7.clicked.connect(self.check_box)
        self.checkBox_9.clicked.connect(self.check_dot)

    def back(self):
        self.hide()
        self.parent.show()

    def done(self):

        with open('settings.py', 'rb') as f:
            data = pickle.load(f)
            data['MODULE_SETTINGS']['graphs'].update(self.settings)

        with open('settings.py', 'wb') as f:
            pickle.dump(data, f)
        self.close()
        # self.hide()
        # self.child.show()


    def check_linear(self):
        if self.checkBox_2.isChecked():
            self.checkBox_2.setChecked(True)
            self.settings['linear'] = True
        else:
            self.checkBox_2.setChecked(False)
            self.settings['linear'] = False

    def check_log(self):
        if self.checkBox_4.isChecked():
            self.checkBox_4.setChecked(True)
            self.settings['log'] = True
        else:
            self.checkBox_4.setChecked(False)
            self.settings['log'] = False

    def check_corr(self):
        if self.checkBox_6.isChecked():
            self.checkBox_6.setChecked(True)
            self.settings['corr'] = True
        else:
            self.checkBox_6.setChecked(False)
            self.settings['corr'] = False

    def check_heatmap(self):
        if self.checkBox_3.isChecked():
            self.checkBox_3.setChecked(True)
            self.settings['heatmap'] = True
        else:
            self.checkBox_3.setChecked(False)
            self.settings['heatmap'] = False

    def check_scatter(self):
        if self.checkBox_8.isChecked():
            self.checkBox_8.setChecked(True)
            self.settings['scatter'] = True
        else:
            self.checkBox_8.setChecked(False)
            self.settings['scatter'] = False

    def check_hist(self):
        if self.checkBox_5.isChecked():
            self.checkBox_5.setChecked(True)
            self.settings['hist'] = True
        else:
            self.checkBox_5.setChecked(False)
            self.settings['hist'] = False

    def check_box(self):
        if self.checkBox_7.isChecked():
            self.checkBox_7.setChecked(True)
            self.settings['box'] = True
        else:
            self.checkBox_7.setChecked(False)
            self.settings['box'] = False

    def check_dot(self):
        if self.checkBox_9.isChecked():
            self.checkBox_9.setChecked(True)
            self.settings['dotplot'] = True
        else:
            self.checkBox_9.setChecked(False)
            self.settings['dotplot'] = False

    def check_bar(self):
        if self.checkBox.isChecked():
            self.checkBox.setChecked(True)
            self.settings['barchart'] = True
        else:
            self.checkBox_9.setChecked(False)
            self.settings['barchart'] = False
