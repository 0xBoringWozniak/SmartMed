import pickle

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import (QWidget, QToolTip, QPushButton, QApplication, QMessageBox)

from .StartingWindow import *
from ..StatisticsApp.StatisticsAppController import StatisticsApp

# logging decorator
import sys
sys.path.append("...")
from logs.logger import debug


class WrappedStartingWindow(StartingWindow, QtWidgets.QMainWindow):

    def __init__(self):

        #self.settings = {}
        super().__init__()
        self.setupUi(self)
        self.__build_buttons()
        self.setWindowTitle('SmartMedProject')


    def __build_buttons(self):
        # create button and add signals
        self.pushButtonStat.clicked.connect(self.button_stats)
        self.pushButtonPred.clicked.connect(self.button_prediction)
        self.pushButtonBioeq.clicked.connect(self.button_bioeq)
        self.pushButtonDone.clicked.connect(self.done)
        self.pushButtonStatQ.clicked.connect(self.stat_q)
        self.pushButtonBiQ.clicked.connect(self.bi_q)
        self.pushButton_PredQ.clicked.connect(self.pred_q)

    def done(self):
        self.close()


    def stat_q(self):
        msg = QMessageBox()
        #third param - width from left, first - lenght from ceiling
        #msg.setGeometry(QtCore.QRect(500, 500, 500, 40))
        msg.setIcon(QMessageBox.Information)
        msg.setInformativeText('Описательный анализ подразумевает получение обощенной информации о данных')
        msg.exec_()

    def bi_q(self):
        msg = QMessageBox()
        #third param - width from left, first - lenght from ceiling
        #msg.setGeometry(QtCore.QRect(500, 500, 500, 40))
        msg.setIcon(QMessageBox.Information)
        msg.setInformativeText('Биоэквивалентность проводит исследование идентичности свойств биодоступности у исходного аппарата и дженерика')
        msg.exec_()

    def pred_q(self):
        msg = QMessageBox()
        #third param - width from left, first - lenght from ceiling
        #msg.setGeometry(QtCore.QRect(500, 500, 500, 40))
        msg.setIcon(QMessageBox.Information)
        msg.setInformativeText('Предсказательный анализ выполняет прогнозирование на основе наколпенной информации')
        msg.exec_()


    def button_stats(self):
        self.hide()
        app = StatisticsApp(menu_window=self)
        app.start()

        # update settings
        return app.settings

    def button_prediction(self):
        self.settings['MODULE'] = 'PREDICTION'

    def button_bioeq(self):
        self.settings['MODULE'] = 'BIOEQ'
