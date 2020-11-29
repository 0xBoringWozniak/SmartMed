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
        self.setWindowTitle('Что-то там')
        
        self.checkBoxFeatures.setChecked(True)
        self.checkBoxDistrub.setChecked(True)
        self.checkBoxPowers.setChecked(True)

        self.settings = {'tables' : {'features': 'True',
                                    'distrub': 'True',
                                    'powers': 'True'}}


    def __build_buttons(self):
        self.pushButtonNext.clicked.connect(self.next)
        self.pushButtonBack.clicked.connect(self.back)
        self.checkBoxFeatures.clicked.connect(self.features)
        self.checkBoxDistrub.clicked.connect(self.distrub)
        self.checkBoxPowers.clicked.connect(self.powers)

    def features(self):
        if self.checkBoxFeatures.isChecked():
            self.checkBoxFeatures.setChecked(True)
            self.settings['tables']['features'] = True
        else:
            self.checkBoxFeatures.setChecked(False)
            self.settings['tables']['features'] = False

    def distrub(self):
        if self.checkBoxDistrub.isChecked():
            self.checkBoxDistrub.setChecked(True)
            self.settings['tables']['distrub'] = True
        else:
            self.checkBoxDistrub.setChecked(False)
            self.settings['tables']['distrub'] = False

    def powers(self):
        if self.checkBoxFeatures.isChecked():
            self.checkBoxFeatures.setChecked(True)
            self.settings['tables']['powers'] = True
        else:
            self.checkBoxFeatures.setChecked(False)
            self.settings['tables']['powers'] = False

    def back(self):
        self.hide()
        self.parent.show()

    def next(self):

        with open('settings.py', 'rb') as f:
            settings = pickle.load(f)
        settings['MODULE_SETTINGS'].update(self.settings)
        with open('settings.py', 'wb') as f:
            pickle.dump(settings, f)
        self.hide()
        self.child.show()


