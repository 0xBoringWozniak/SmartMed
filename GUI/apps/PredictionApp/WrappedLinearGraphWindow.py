import pickle

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (
    QWidget, QToolTip, QPushButton, QApplication, QMessageBox, )

from .LinearGraphWindow import LinearGraphWindow


class WrappedLinearGraphWindow(LinearGraphWindow, QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.__build_buttons()
        self.setWindowTitle('Выбор графиков')
        self.checkBoxEq.setChecked(True)
        self.checkBoxQuality.setChecked(True)
        self.checkBoxResid.setChecked(True)
        self.checkBoxSignif.setChecked(True)
        self.checkBoxDistribResid.setChecked(True)
        self.settings = {'distrib_resid': True,
                         'equation': True,
                         'model_quality': True,
                         'resid': True,
                         'signif': True}
       

    def __build_buttons(self):
        self.pushButtonDone.clicked.connect(self.done)
        self.pushButtonBack.clicked.connect(self.back)
        self.checkBoxDistribResid.clicked.connect(self.distr_resid)
        self.checkBoxEq.clicked.connect(self.eq)
        self.checkBoxQuality.clicked.connect(self.quality)
        self.checkBoxResid.clicked.connect(self.resid)
        self.checkBoxSignif.clicked.connect(self.sign)

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

    def distr_resid(self):
        if self.checkBoxDistribResid.isChecked():
            self.checkBoxDistribResid.setChecked(True)
            self.settings['distrib_resid'] = True
        else:
            self.checkBoxDistribResid.setChecked(False)
            self.settings['distrib_resid'] = False

    def eq(self):
        if self.checkBoxEq.isChecked():
            self.checkBoxEq.setChecked(True)
            self.settings['equation'] = True
        else:
            self.checkBoxEq.setChecked(False)
            self.settings['equation'] = False

    def quality(self):
        if self.checkBoxQuality.isChecked():
            self.checkBoxQuality.setChecked(True)
            self.settings['model_quality'] = True
        else:
            self.checkBoxQuality.setChecked(False)
            self.settings['model_quality'] = False

    def resid(self):
        if self.checkBoxResid.isChecked():
            self.checkBoxResid.setChecked(True)
            self.settings['resid'] = True
        else:
            self.checkBoxResid.setChecked(False)
            self.settings['resid'] = False

    def sign(self):
        if self.checkBoxSignif.isChecked():
            self.checkBoxSignif.setChecked(True)
            self.settings['signif'] = True
        else:
            self.checkBoxSignif.setChecked(False)
            self.settings['signif'] = False
