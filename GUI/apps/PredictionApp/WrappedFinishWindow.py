import pickle
import threading
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (
    QWidget, QToolTip, QPushButton, QApplication, QMessageBox, )

from .FinishWindow import FinishWindow
from backend import ModuleManipulator


class WrappedFinishWindow(FinishWindow, QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.__build_buttons()
        self.setWindowTitle('Что-то там 3')

    def __build_buttons(self):
        self.pushButtonDone.clicked.connect(self.done)
        self.pushButtonBack.clicked.connect(self.back)

    def back(self):
        self.hide()
        self.parent.show()

    def done(self):

        with open('settings.py', 'rb') as f:
            settings = pickle.load(f)
        module_starter = ModuleManipulator(settings)
        threading.Thread(target=module_starter.start, daemon=True).start()

        self.close()
