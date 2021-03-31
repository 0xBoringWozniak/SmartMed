import pickle

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (
    QWidget, QToolTip, QPushButton, QApplication, QMessageBox)

from .TablesWindow import TablesWindow



class WrappedTablesWindow(TablesWindow, QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.__build_buttons()
        self.setWindowTitle('Результаты')
        
        self.checkBoxDistrub.setChecked(True)
        self.checkBoxFeatures.setChecked(True)
        self.checkBoxPowers.setChecked(True)
        self.checkBoxRes.setChecked(True)

        self.settings = {'tables' : {'criteria': 'True',
                                    'features': 'True',
                                    'var': 'True'}}


    def __build_buttons(self):
        self.pushButtonNext.clicked.connect(self.next)
        self.pushButtonBack.clicked.connect(self.back)
        self.checkBoxFeatures.clicked.connect(self.features)
        self.checkBoxDistrub.clicked.connect(self.distrub)
        self.checkBoxPowers.clicked.connect(self.powers)

    def features(self):
        if self.checkBoxFeatures.isChecked():
            self.checkBoxFeatures.setChecked(True)
            self.settings['tables']['criteria'] = True
        else:
            self.checkBoxFeatures.setChecked(False)
            self.settings['tables']['criteria'] = False

    def distrub(self):
        if self.checkBoxDistrub.isChecked():
            self.checkBoxDistrub.setChecked(True)
            self.settings['tables']['features'] = True
        else:
            self.checkBoxDistrub.setChecked(False)
            self.settings['tables']['features'] = False

    def powers(self):
        if self.checkBoxPowers.isChecked():
            self.checkBoxPowers.setChecked(True)
            self.settings['tables']['var'] = True
        else:
            self.checkBoxPowers.setChecked(False)
            self.settings['tables']['var'] = False

    def back(self):
        self.hide()
        self.parent_parral.show()

    def next(self):

        with open('settings.py', 'rb') as f:
            settings = pickle.load(f)
        settings['MODULE_SETTINGS'].update(self.settings)
        with open('settings.py', 'wb') as f:
            pickle.dump(settings, f)
        self.hide()
        self.child_parral.show()


