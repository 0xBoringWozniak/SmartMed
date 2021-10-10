import pickle

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (
    QWidget, QToolTip, QPushButton, QApplication, QMessageBox)

from .TablesWindowCross import TablesWindowCross


class WrappedTablesWindowCross(TablesWindowCross, QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.__build_buttons()
        self.setWindowTitle('Результаты')

        self.checkBoxFeatures.setChecked(True)
        self.checkBoxCriteria.setChecked(True)
        self.checkBox.setChecked(True)
        self.checkBoxStat.setChecked(True)

        self.settings = {'tables': {'avg_auc': True,
                                    'anal_resylts': True,
                                    'results': True,
                                    'statistics' : True}}

    def __build_buttons(self):
        self.pushButtonNext.clicked.connect(self.next)
        self.pushButtonBack.clicked.connect(self.back)
        self.checkBoxCriteria.clicked.connect(self.features)
        self.checkBoxFeatures.clicked.connect(self.distrub)
        self.checkBox.clicked.connect(self.res)
        self.checkBoxStat.clicked.connect(self.statistics)

    def features(self):
        if self.checkBoxCriteria.isChecked():
            self.checkBoxCriteria.setChecked(True)
            self.settings['tables']['avg_auc'] = True
        else:
            self.checkBoxCriteria.setChecked(False)
            self.settings['tables']['avg_auc'] = False

    def distrub(self):
        if self.checkBoxFeatures.isChecked():
            self.checkBoxFeatures.setChecked(True)
            self.settings['tables']['anal_resylts'] = True
        else:
            self.checkBoxFeatures.setChecked(False)
            self.settings['tables']['anal_resylts'] = False

    def res(self):
        if self.checkBox.isChecked():
            self.checkBox.setChecked(True)
            self.settings['tables']['results'] = True
        else:
            self.checkBox.setChecked(False)
            self.settings['tables']['results'] = False


    def statistics(self):
        if self.checkBoxStat.isChecked():
            self.checkBoxStat.setChecked(True)
            self.settings['tables']['statistics'] = True
        else:
            self.checkBoxStat.setChecked(False)
            self.settings['tables']['statistics'] = False

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
