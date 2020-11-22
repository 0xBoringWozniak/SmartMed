import pickle
import pandas as pd

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (
    QWidget, QToolTip, QPushButton, QApplication, QMessageBox, )

from .DownloadWindow import DownloadWindow


class WrappedDownloadWindow(DownloadWindow, QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.__build_buttons()
        self.setWindowTitle('Что-то там 2')
        self.columns =''
       

    def __build_buttons(self):
        #плохо с неймингом, надо переделать бек некст
        self.pushButtonBack.clicked.connect(self.back
            )
        self.pushButtonNext.clicked.connect(self.next)
        self.pushButton.clicked.connect(self.path_to_file)
    def back(self):
        self.hide()
        self.parent.show()

    def next(self):
        
        self.hide()
        self.child.show()

    def path_to_file(self):
         data = QtWidgets.QFileDialog.getOpenFileName()[0]

         self.columns = pd.read_excel(data).columns
         print(list(self.columns))
         self.comboBox.addItems(self.columns)