import pickle

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (
    QWidget, QToolTip, QPushButton, QApplication, QMessageBox, QTableWidget)
from .RadioWindow import RadioWindow


class WrappedRadioWindow(RadioWindow, QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.__build_buttons()
        self.setWindowTitle('Предобработка данных')
        self.comboBox.addItems(["Средним/модой (численные/категориальные значения)",
                                "Введенным значением (требуется ввод для каждого столбца отдельно)",
                                "Удаление строк с пропущенными значениями",
                                "Медианной/модой (численные/категориальные значения)"
                                ])
        self.settings = {'preprocessing': {
            'fillna': 'mean',
            'encoding': 'label_encoding',
            'scaling': False
        }
        }

    def __build_buttons(self):
        self.pushButtonNext.clicked.connect(self.next)
        self.pushButtonBack.clicked.connect(self.back)

    def back(self):
        self.hide()
        self.parent.show()

    def next(self):
        value_na = self.comboBox.currentText()
        if value_na == 'средним/модой (аналогично)':
            self.settings['preprocessing'] = 'mean'
        elif value_na == 'заданным значием (требуется ввод для каждого столбца отдельно)':
            self.settings['preprocessing'] = 'exact_value'
        elif value_na == 'откидывание строк с пропущенными значениями':
            self.settings['preprocessing'] = 'dropna'
        else:
            self.settings['preprocessing'] = 'median'
        with open('settings.py', 'rb') as f:
            data = pickle.load(f)
        data['MODULE_SETTINGS'].update(self.settings)
        with open('settings.py', 'wb') as f:
            pickle.dump(data, f)
        self.hide()
        self.child.show()
