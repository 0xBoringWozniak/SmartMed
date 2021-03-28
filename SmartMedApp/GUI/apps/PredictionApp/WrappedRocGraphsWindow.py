import pickle
import threading

from PyQt5.QtCore import QTimer, QEventLoop
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (
    QWidget, QToolTip, QPushButton, QApplication, QMessageBox, )

from .RocGraphsWindow import RocGraphsWindow
from ..utils import remove_if_exists
from ..WaitingSpinnerWidget import QtWaitingSpinner

from SmartMedApp.backend import ModuleManipulator


class WrappedRocGraphsWindow(RocGraphsWindow, QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.settings = {'points_table': True,
                         'metrics_table': True,
                         'spec_and_sens': True,
                         'spec_and_sens_table': True,
                         'classificators_comparison': True}
        self.checkBox.setChecked(True)
        self.checkBox_2.setChecked(True)
        self.checkBox_3.setChecked(True)
        self.checkBox_4.setChecked(True)
        self.checkBox_9.setChecked(True)
        self.__build_buttons()
        self.setWindowTitle(' ')

    def __build_buttons(self):
        self.pushButtonDone.clicked.connect(self.done)
        self.pushButtonBack.clicked.connect(self.back)

    def back(self):
        self.hide()
        self.parent.show()

    def done(self):
        if self.checkBox_2.isChecked() != True:
            self.settings['points_table'] = False
        if self.checkBox_4.isChecked() != True:
            self.settings['metrics_table'] = False
        if self.checkBox.isChecked() != True:
            self.settings['spec_and_sens'] = False
        if self.checkBox_3.isChecked() != True:
            self.settings['spec_and_sens_table'] = False
        if self.checkBox_9.isChecked() != True:
            self.settings['classificators_comparison'] = False
        with open('settings.py', 'rb') as f:
            data = pickle.load(f)
        data['MODULE_SETTINGS'].update(self.settings)
        data['MODULE_SETTINGS'].pop('columns')
        with open('settings.py', 'wb') as f:
            pickle.dump(data, f)
        module_starter = ModuleManipulator(data)
        threading.Thread(target=module_starter.start, daemon=True).start()
        self.spinner = QtWaitingSpinner(self)
        self.layout().addWidget(self.spinner)
        self.spinner.start()
        #QTimer.singleShot(10000, self.spinner.stop)
        loop = QEventLoop()
        QTimer.singleShot(10000, loop.quit)
        loop.exec_()
        self.spinner.stop()
        self.close()
        self.child.show()
        print(data)
        remove_if_exists()
