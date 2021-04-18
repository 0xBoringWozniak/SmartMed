import pickle
import threading

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (
    QWidget, QToolTip, QPushButton, QApplication, QMessageBox)

from .GraphsWindow import GraphsWindow
from ..WaitingSpinnerWidget import QtWaitingSpinner
from PyQt5.QtCore import QTimer, QEventLoop
from ..utils import remove_if_exists

from SmartMedApp.backend import ModuleManipulator


class WrappedGraphsWindow(GraphsWindow, QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.__build_buttons()
        self.setWindowTitle('Визуализация')

        self.checkBoxAllinGroup.setChecked(True)
        self.checkBoxLogAllinGroup.setChecked(True)
        self.checkBoxForEachGroup.setChecked(True)
        self.checkBoxLogForEachGroup.setChecked(True)

        self.settings = {'graphs': {'all_in_group': True,
                                    'log_all_in_group': True,
                                    'each_in_group': True,
                                    'log_each_in_group': True}}

    def __build_buttons(self):
        self.pushButtonNext.clicked.connect(self.next)
        self.pushButtonBack.clicked.connect(self.back)
        self.checkBoxAllinGroup.clicked.connect(self.all_in_group)
        self.checkBoxLogAllinGroup.clicked.connect(self.log_all_in_group)
        self.checkBoxForEachGroup.clicked.connect(self.each_in_group)
        self.checkBoxLogForEachGroup.clicked.connect(self.log_each_in_group)

    def all_in_group(self):
        if self.checkBoxAllinGroup.isChecked():
            self.checkBoxAllinGroup.setChecked(True)
            self.settings['graphs']['all_in_group'] = True
        else:
            self.checkBoxAllinGroup.setChecked(False)
            self.settings['graphs']['all_in_group'] = False

    def log_all_in_group(self):
        if self.checkBoxLogAllinGroup.isChecked():
            self.checkBoxLogAllinGroup.setChecked(True)
            self.settings['graphs']['log_all_in_group'] = True
        else:
            self.checkBoxLogAllinGroup.setChecked(False)
            self.settings['graphs']['log_all_in_group'] = False

    def each_in_group(self):
        if self.checkBoxForEachGroup.isChecked():
            self.checkBoxForEachGroup.setChecked(True)
            self.settings['graphs']['each_in_group'] = True
        else:
            self.checkBoxForEachGroup.setChecked(False)
            self.settings['graphs']['each_in_group'] = False

    def log_each_in_group(self):
        if self.checkBoxLogForEachGroup.isChecked():
            self.checkBoxLogForEachGroup.setChecked(True)
            self.settings['graphs']['log_each_in_group'] = True
        else:
            self.checkBoxLogForEachGroup.setChecked(False)
            self.settings['graphs']['log_each_in_group'] = False

    def back(self):
        self.hide()
        self.parent_parral.show()

    def next(self):
        with open('settings.py', 'rb') as f:
            settings = pickle.load(f)
        settings['MODULE_SETTINGS'].update(self.settings)
        with open('settings.py', 'wb') as f:
            pickle.dump(settings, f)

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
        self.hide()
        self.child_parral.show()
        remove_if_exists()
