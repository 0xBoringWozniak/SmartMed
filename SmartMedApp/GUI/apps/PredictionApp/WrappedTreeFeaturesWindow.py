import pickle
import threading

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (
    QWidget, QToolTip, QPushButton, QApplication, QMessageBox, )

from .TreeFeaturesWindow import TreeFeaturesWindow


class WrappedTreeFeaturesWindow(TreeFeaturesWindow, QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.__build_buttons()
        self.settings = {'sort': True}
        self.checkBox.setChecked(True)

    def __build_buttons(self):
        self.pushButtonNext.clicked.connect(self.next)
        self.pushButtonBack.clicked.connect(self.back)
        self.checkBox.clicked.connect(self.sort)
        self.setWindowTitle(' ')

    def back(self):
        self.hide()
        self.parent.show()

    def sort(self):
        if self.checkBox.isChecked():
            self.checkBox.setChecked(True)
            self.settings['sort'] = True
        else:
            self.checkBox.setChecked(False)
            self.settings['sort'] = False

    def next(self):
        depth = self.lineEdit.text()
        min_sample_number = self.lineEdit_2.text()
        features_count = self.lineEdit_3.text()
        with open('settings.py', 'rb') as f:
            data = pickle.load(f)
            data['MODULE_SETTINGS'].update({'tree_depth': depth,
                                            'samples': min_sample_number,
                                            'features_count': features_count})
            data['MODULE_SETTINGS'].update(self.settings)
        with open('settings.py', 'wb') as f:
            pickle.dump(data, f)
        self.hide()
        self.child.show()
