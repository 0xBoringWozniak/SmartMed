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
        self.setWindowTitle('Загрузка данных')
        self.settings = {'path_test': '', 'path_ref': '' }
  
    def __build_buttons(self):
        # плохо с неймингом, надо переделать бек некст
        self.pushButtonNext.clicked.connect(self.next)
        self.pushButtonBack.clicked.connect(self.back)
        self.pushButtonDownload.clicked.connect(self.download)
        self.pushButtonDownload1.clicked.connect(self.download1)

    def back(self):
        self.hide()
        self.parent_parral.show()

    def next(self):
        while self.settings['path_test'] == '' or self.settings['path_ref'] == '':
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Error")
            msg.setInformativeText('Please, choose path to file')
            msg.setWindowTitle("Error")
            msg.exec_()

            return 
        with open('settings.py', 'rb') as f:
            data = pickle.load(f)
            data['MODULE_SETTINGS'].update(self.settings)
        with open('settings.py', 'wb') as f:
            pickle.dump(data, f)

        self.hide()
        self.child_parral.show()


    def download(self):
        self.settings['path_test'] = QtWidgets.QFileDialog.getOpenFileName()[0]


    def download1(self):
        self.settings['path_ref'] = QtWidgets.QFileDialog.getOpenFileName()[0]
