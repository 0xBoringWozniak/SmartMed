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
        
        self.settings = {'tables' : {'avg_auc': 'True',
                                    'anal_resylts': 'True',
                                    }}


    def __build_buttons(self):
        self.pushButtonNext.clicked.connect(self.next)
        self.pushButtonBack.clicked.connect(self.back)

        #self.checkBoxСriteria.clicked.connect(self.features)
        self.checkBoxFeatures.clicked.connect(self.distrub)

    def features(self):
        if self.checkBoxСriteria.isChecked():
            self.checkBoxСriteria.setChecked(True)
            self.settings['tables']['avg_auc'] = True
        else:
            self.checkBoxСriteria.setChecked(False)
            self.settings['tables']['avg_auc'] = False

    def distrub(self):
        if self.checkBoxFeatures.isChecked():
            self.checkBoxFeatures.setChecked(True)
            self.settings['tables']['anal_resylts'] = True
        else:
            self.checkBoxFeatures.setChecked(False)
            self.settings['tables']['anal_resylts'] = False


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

