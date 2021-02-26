import pickle
import threading

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (
    QWidget, QToolTip, QPushButton, QApplication, QMessageBox)

from .UniformityWindow import UniformityWindow
from SmartMedApp.backend import ModuleManipulator
from ..WaitingSpinnerWidget import QtWaitingSpinner
from PyQt5.QtCore import QTimer, QEventLoop
from ..utils import remove_if_exists


class WrappedUniformityWindow(UniformityWindow, QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.__build_buttons()
        self.radioButtonF.setChecked(True)
        #self.setWindowTitle('Что-то там')
    

    def __build_buttons(self):
        self.pushButtonDone.clicked.connect(self.done)
        self.pushButtonBack.clicked.connect(self.back)
        #self.radioButton_cross.clicked.connect(self.cross)
        #self.radioButton_parall.clicked.connect(self.parall)

    def back(self):
        self.hide()
        self.parent_parral.show()

    def done(self):
        with open('settings.py', 'rb') as f:
            settings = pickle.load(f)
        if self.radioButtonF.isChecked():
            settings['MODULE_SETTINGS']['uniformity'] = 'F'
        else:
            settings['MODULE_SETTINGS']['uniformity'] = 'Leven'
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

