import pickle
import threading
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (
    QWidget, QToolTip, QPushButton, QApplication, QMessageBox, )

from .RocGraphsWindow import RocGraphsWindow


class WrappedRocGraphsWindow(RocGraphsWindow, QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.settings = {'points_table': True,
                        'metrics_table': True,
                        'spec_and_sens': True,
                        'spec_and_sens_table':True}
        self.__build_buttons()

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
        with open('settings.py', 'rb') as f:
            data = pickle.load(f)
        data['MODULE_SETTINGS'].update(self.settings)
        data['MODULE_SETTINGS'].pop('columns')
        with open('settings.py', 'wb') as f:
            pickle.dump(data, f)
        self.close()
        self.child.show()
        print(data)
        
