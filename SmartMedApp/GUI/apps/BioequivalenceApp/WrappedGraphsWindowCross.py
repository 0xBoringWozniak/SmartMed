import pickle

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (
    QWidget, QToolTip, QPushButton, QApplication, QMessageBox)

from .GraphsWindowCross import GraphsWindowCross
from SmartMedApp.backend import ModuleManipulator
from ..WaitingSpinnerWidget import QtWaitingSpinner
from PyQt5.QtCore import QTimer, QEventLoop
from ..utils import remove_if_exists
import threading


class WrappedGraphsWindowCross(GraphsWindowCross, QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.__build_buttons()
        self.setWindowTitle('Графики')

        self.checkBoxAllinGroup.setChecked(True)
        self.checkBoxLogAllinGroup.setChecked(True)
        self.checkBoxForEachGroup.setChecked(True)

        self.settings = {'graphs' : {'indiv_concet': True,
                                    'avg_concet': True,
                                    'gen_concet': True}}



    def __build_buttons(self):
        self.pushButtonNext.clicked.connect(self.next)
        self.pushButtonBack.clicked.connect(self.back)
        self.checkBoxAllinGroup.clicked.connect(self.indiv_concet)
        self.checkBoxLogAllinGroup.clicked.connect(self.avg_concet)
        self.checkBoxForEachGroup.clicked.connect(self.gen_concet)

    def indiv_concet(self):
        if self.checkBoxAllinGroup.isChecked():
            self.checkBoxAllinGroup.setChecked(True)
            self.settings['graphs']['indiv_concet'] = True
        else:
            self.checkBoxAllinGroup.setChecked(False)
            self.settings['graphs']['indiv_concet']  = False

    def avg_concet(self):
        if self.checkBoxLogAllinGroup.isChecked():
            self.checkBoxLogAllinGroup.setChecked(True)
            self.settings['graphs']['avg_concet'] = True
        else:
            self.checkBoxLogAllinGroup.setChecked(False)
            self.settings['graphs']['avg_concet']  = False


    def gen_concet(self):
        if self.checkBoxForEachGroup.isChecked():
            self.checkBoxForEachGroup.setChecked(True)
            self.settings['graphs']['gen_concet'] = True
        else:
            self.checkBoxForEachGroup.setChecked(False)
            self.settings['graphs']['gen_concet']  = False

    def back(self):
        self.hide()
        self.parent_cross.show()

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
        self.child_cross.show()
        remove_if_exists()

