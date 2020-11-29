import pickle

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (
    QWidget, QToolTip, QPushButton, QApplication, QMessageBox, )

from .DownloadWindow import DownloadWindow


class WrappedDownloadWindow(DownloadWindow, QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.__build_buttons()
        self.setWindowTitle('Что-то там')
        self.settings = { 'MODULE': 'BIOEQ', 'MODULE_SETTINGS': {'path': '' }}
  
    def __build_buttons(self):
        # плохо с неймингом, надо переделать бек некст
        self.pushButtonBack.clicked.connect(self.next)
        self.pushButtonNext.clicked.connect(self.back)
        self.pushButtondDownload.clicked.connect(self.download)

    def back(self):
        self.hide()
        self.parent.show()

    def next(self):
        while self.settings['MODULE_SETTINGS']['path'] == '':
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Error")
            msg.setInformativeText('Please, choose path to file')
            msg.setWindowTitle("Error")
            msg.exec_()
            return 
        
        with open('settings.py', 'wb') as f:
            pickle.dump(self.settings, f)

        self.hide()
        self.child.show()


    def download(self):
        self.settings['MODULE_SETTINGS']['path'] = QtWidgets.QFileDialog.getOpenFileName()[0]
