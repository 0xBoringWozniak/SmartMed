import sys

from PyQt5 import QtWidgets

from .apps import StartingApp

# logging decorator
from SmartMedApp.logs.logger import debug


class GUI:
    '''Qt apps manipulator'''

    def __init__(self):
        '''create QApp'''
        self.__make_QAapp()

    @debug
    def __make_QAapp(self):
        self.mainQt = QtWidgets.QApplication(sys.argv)

    @debug
    def start_gui(self):
        '''display QT app'''
        app = StartingApp()
        app.start()
        self.mainQt.exec_()
