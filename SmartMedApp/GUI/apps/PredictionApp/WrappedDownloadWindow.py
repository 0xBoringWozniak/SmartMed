import pickle
import pandas as pd

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget, QToolTip, QPushButton, QApplication, QMessageBox)

from .DownloadWindow import DownloadWindow
from ..Notification import NotificationWindow
from ..utils import get_columns, remove_if_exists

from SmartMedApp.backend.modules.dataprep import PandasPreprocessor


class WrappedDownloadWindow(DownloadWindow, QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.__build_buttons()
        self.setWindowTitle('Загрузка данных')
        self.settings = {'MODULE': 'PREDICT', 'MODULE_SETTINGS': {'path': '', 'columns': '',
                                                                  'preprocessing': '', 'model': ''
                                                                  }}
        self.columns = ''

    def __build_buttons(self):
        # плохо с неймингом, надо переделать бек некст
        self.pushButtonBack.clicked.connect(self.back)
        self.pushButtonNext.clicked.connect(self.next)
        self.pushButton.clicked.connect(self.path_to_file)

    def back(self):
        remove_if_exists()
        self.hide()
        self.parent.show()

    def next(self):
        while self.settings['MODULE_SETTINGS']['path'] == '':
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Ошибка")
            msg.setInformativeText('Выберите файл')
            msg.setWindowTitle("Ошибка")
            msg.exec_()
            return
        self.hide()
        self.child.show()

    def path_to_file(self):
        path = QtWidgets.QFileDialog.getOpenFileName()[0]
        if path != '':
            self.settings['MODULE_SETTINGS']['path'] = path
            #self.prep = PandasPreprocessor(self.settings['MODULE_SETTINGS']).__read_file().columns
            # print(self.prep)
            # self.prep._read_file()
            self.settings['MODULE_SETTINGS'][
                'columns'] = get_columns(path).columns
            #self.settings['MODULE_SETTINGS']['columns'] = prep.df.columns
        with open('settings.py', 'wb') as f:
            pickle.dump(self.settings, f)

           # self.comboBox.addItems(col)
