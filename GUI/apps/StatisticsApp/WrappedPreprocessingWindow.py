import pickle

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import (
    QWidget, QToolTip, QPushButton, QApplication, QMessageBox)

from .PreprocessingWindow import PreprocessingWindow


class WrappedPreprocessingWindow(PreprocessingWindow, QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.settings = {'MODULE_SETTINGS': {
            'metrics': {}, 'graphs': {}}, 'MODULE': 'STATS'}
        self.settings['MODULE_SETTINGS']['data'] = {'preprocessing': {
            'fillna': 'mean',
            'encoding': 'label_encoding',
            'scaling': False
        },
            'path': ''
        }
        self.__build_buttons()
        self.setWindowTitle('Препроцессинг')
        self.comboBox1.addItems(["средним/модой (аналогично)",
                                 "заданным значием (требуется ввод для каждого столбца отдельно)",
                                 "откидывание строк с пропущенными значениями",
                                 "медианным/модой (численные/категориальные соответсвенно)"
                                 ])

        self.comboBox2.addItems(["label encoding", "dummy encoding",
                                 "one hot encoding", "binary encoding"])

    def __build_buttons(self):
        self.pushButtonNext.clicked.connect(self.next)
        self.pushButtonBack.clicked.connect(self.back)
        self.commandLinkButton.clicked.connect(self.path_to_file)

    def back(self):
        self.hide()
        self.parent.show()

    def next(self):
        while self.settings['MODULE_SETTINGS']['data']['path'] == '':
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Error")
            msg.setInformativeText('Please, choose path to file')
            msg.setWindowTitle("Error")
            msg.exec_()
            return 
        value_na = self.comboBox1.currentText()
        value_encoding = self.comboBox2.currentText()

        if value_na == 'средним/модой (аналогично)':
            self.settings['MODULE_SETTINGS']['data']['fillna'] = 'mean'
        elif value_na == 'заданным значием (требуется ввод для каждого столбца отдельно)':
            self.settings['MODULE_SETTINGS']['data']['fillna'] = 'exact_value'
        elif value_na == 'откидывание строк с пропущенными значениями':
            self.settings['MODULE_SETTINGS']['data']['fillna'] = 'dropna'
        else:
            self.settings['MODULE_SETTINGS']['data']['fillna'] = 'median'
        if value_encoding == 'label encoding':
            self.settings['MODULE_SETTINGS']['data'][
                'encoding'] = 'label_encoding'
        elif value_encoding == 'dummy encoding':
            self.settings['MODULE_SETTINGS']['data'][
                'encoding'] = 'dummy encoding'
        elif value_encoding == 'one hot encoding':
            self.settings['MODULE_SETTINGS']['data'][
                'encoding'] = 'one hot encoding'
        else:
            self.settings['MODULE_SETTINGS']['data'][
                'encoding'] = 'binary encoding'

        with open('settings.py', 'wb') as f:
            pickle.dump(self.settings, f)

        self.hide()
        self.child.show()

    def path_to_file(self):

        self.settings['MODULE_SETTINGS']['data']['path'] = QtWidgets.QFileDialog.getOpenFileName()[
            0]
