import pickle
import threading
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (
    QWidget, QToolTip, QPushButton, QApplication, QMessageBox, )

from .TreeVisualWindow import TreeVisualWindow


class WrappedTreeVisualWindow(TreeVisualWindow, QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.__build_buttons()
        self.settings = {'tree': True,
                         'table': True,
                         'indicators': True}
        self.checkBoxTree.setChecked(True)
        self.checkBoxTablr.setChecked(True)
        self.checkBoxValue.setChecked(True)

    def __build_buttons(self):
        self.pushButtonDone.clicked.connect(self.done)
        self.pushButtonBack.clicked.connect(self.back)
        self.checkBoxTree.clicked.connect(self.tree)
        self.checkBoxTablr.clicked.connect(self.table)
        self.checkBoxValue.clicked.connect(self.indicators)

    def tree(self):
        if self.checkBoxTree.isChecked():
            self.checkBoxTree.setChecked(True)
            self.settings['tree'] = True
        else:
            self.checkBoxTree.setChecked(False)
            self.settings['tree'] = False

    def table(self):
        if self.checkBoxTablr.isChecked():
            self.checkBoxTablr.setChecked(True)
            self.settings['table'] = True
        else:
            self.checkBoxTablr.setChecked(False)
            self.settings['table'] = False

    def indicators(self):
        if self.checkBoxValue.isChecked():
            self.checkBoxValue.setChecked(True)
            self.settings['indicators'] = True
        else:
            self.checkBoxValue.setChecked(False)
            self.settings['indicators'] = False



    def back(self):
        self.hide()
        self.parent.show()

    def done(self):
        with open('settings.py', 'rb') as f:
            data = pickle.load(f)
            data['MODULE_SETTINGS'].update(self.settings)
            data['MODULE_SETTINGS'].pop('columns')
        with open('settings.py', 'wb') as f:
            pickle.dump(data, f)
        self.close()
        self.child.show()
        print(data)

        
