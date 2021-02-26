import pickle

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (
    QWidget, QToolTip, QPushButton, QApplication, QMessageBox)

from .GraphsWindowCross import GraphsWindowCross



class WrappedGraphsWindowCross(GraphsWindowCross, QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.__build_buttons()
        #self.setWindowTitle('Что-то там')

        self.checkBoxAllinGroup.setChecked(True)
        self.checkBoxLogAllinGroup.setChecked(True)
        self.checkBoxLogForEachGroup.setChecked(True)

        self.settings = {'graphs' : {'indiv_concet': True,
                                    'avg_concet': True,
                                    'gen_concet': True}}



    def __build_buttons(self):
        self.pushButtonNext.clicked.connect(self.next)
        self.pushButtonBack.clicked.connect(self.back)
        self.checkBoxAllinGroup.clicked.connect(self.indiv_concet)
        self.checkBoxLogAllinGroup.clicked.connect(self.avg_concet)
        self.checkBoxLogForEachGroup.clicked.connect(self.gen_concet)

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
        if self.checkBoxLogForEachGroup.isChecked():
            self.checkBoxLogForEachGroup.setChecked(True)
            self.settings['graphs']['gen_concet'] = True
        else:
            self.checkBoxLogForEachGroup.setChecked(False)
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
        self.hide()
        self.child_cross.show()

