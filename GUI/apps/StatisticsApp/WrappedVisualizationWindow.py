import pickle
import time
import os
import threading

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import (
    QWidget, QToolTip, QPushButton, QApplication, QMessageBox)
from PyQt5.QtCore import QTimer, QEventLoop

from .VisualizationWindow import VisualizationWindow
from .WaitingSpinnerWidget import QtWaitingSpinner


import sys
sys.path.append("...")

from backend import ModuleManipulator


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
            'box': True,
            'piechart': True,
            'dotplot': True,
        }
        self.checkBoxPie.setChecked(True)
        self.checkBoxLinear.setChecked(True)
        self.checkBoxHeatmap.setChecked(True)
        self.checkBoxLog.setChecked(True)
        self.checkBoxHist.setChecked(True)
        self.checkBoxCorr.setChecked(True)
        self.checkBoxBox.setChecked(True)
        self.checkBoxScatter.setChecked(True)
        self.checkBoxDot.setChecked(True)
        self.__build_buttons()


    def __build_buttons(self):
        self.pushButton.clicked.connect(self.back)
        self.pushButtonDone.clicked.connect(self.done)
        self.checkBoxPie.clicked.connect(self.check_pie)
        self.checkBoxLinear.clicked.connect(self.check_linear)
        self.checkBoxLog.clicked.connect(self.check_log)
        self.checkBoxCorr.clicked.connect(self.check_corr)
        self.checkBoxHeatmap.clicked.connect(self.check_heatmap)
        self.checkBoxScatter.clicked.connect(self.check_scatter)
        self.checkBoxHist.clicked.connect(self.check_hist)
        self.checkBoxBox.clicked.connect(self.check_box)
        self.checkBoxDot.clicked.connect(self.check_dot)

    def back(self):
        
        self.hide()
        self.hide()
        self.parent.show()

    def done(self):

        with open('settings.py', 'rb') as f:
            data = pickle.load(f)
            data['MODULE_SETTINGS']['graphs'].update(self.settings)

        with open('settings.py', 'wb') as f:
            pickle.dump(data, f)

        with open('settings.py', 'rb') as f:
            settings = pickle.load(f)

        module_starter = ModuleManipulator(settings)
        threading.Thread(target=module_starter.start, daemon=True).start()
        self.spinner = QtWaitingSpinner(self)
        self.layout().addWidget(self.spinner)
        self.spinner.start()
        #QTimer.singleShot(10000, self.spinner.stop)
        loop = QEventLoop()
        QTimer.singleShot(10000, loop.quit)
        loop.exec_()
        self.spinner.stop()
        if os.path.exists('settings.py'):
            os.remove('settings.py')
        self.child.show()

    def check_linear(self):

        if self.checkBoxLinear.isChecked():
            self.checkBoxLinear.setChecked(True)
            self.settings['linear'] = True
        else:
            self.checkBoxLinear.setChecked(False)
            self.settings['linear'] = False

    def check_log(self):
        if self.checkBoxLog.isChecked():
            self.checkBoxLog.setChecked(True)
            self.settings['log'] = True
        else:
            self.checkBoxLog.setChecked(False)
            self.settings['log'] = False

    def check_corr(self):
        if self.checkBoxCorr.isChecked():
            self.checkBoxCorr.setChecked(True)
            self.settings['corr'] = True
        else:
            self.checkBoxCorr.setChecked(False)
            self.settings['corr'] = False

    def check_heatmap(self):
        if self.checkBoxHeatmap.isChecked():
            self.checkBoxHeatmap.setChecked(True)
            self.settings['heatmap'] = True
        else:
            self.checkBoxHeatmap.setChecked(False)
            self.settings['heatmap'] = False

    def check_scatter(self):
        if self.checkBoxScatter.isChecked():
            self.checkBoxScatter.setChecked(True)
            self.settings['scatter'] = True
        else:
            self.checkBoxScatter.setChecked(False)
            self.settings['scatter'] = False

    def check_hist(self):
        if self.checkBoxHist.isChecked():
            self.checkBoxHist.setChecked(True)
            self.settings['hist'] = True
        else:
            self.checkBoxHist.setChecked(False)
            self.settings['hist'] = False

    def check_box(self):
        if self.checkBoxBox.isChecked():
            self.checkBoxBox.setChecked(True)
            self.settings['box'] = True
        else:
            self.checkBoxBox.setChecked(False)
            self.settings['box'] = False

    def check_dot(self):
        if self.checkBoxDot.isChecked():
            self.checkBoxDot.setChecked(True)
            self.settings['dotplot'] = True
        else:
            self.checkBoxDot.setChecked(False)
            self.settings['dotplot'] = False

    def check_pie(self):
        if self.checkBoxPie.isChecked():
            self.checkBoxPie.setChecked(True)
            self.settings['piechart'] = True
        else:
            self.checkBoxPie.setChecked(False)
            self.settings['piechart'] = False
