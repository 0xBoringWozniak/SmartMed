import pickle

from PyQt5 import QtWidgets

from .PrepWindow import RadioWindow


class WrappedRadioWindow(RadioWindow, QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.__build_buttons()
        self.setWindowTitle('Предобработка данных')
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
        if value_na == 'Средним/модой (численные/категориальные значения)':
            self.settings['fillna'] = 'mean'
        elif value_na == 'Введенным значением (требуется ввод для каждого столбца отдельно)':
            self.settings['fillna'] = 'exact_value'
        elif value_na == 'Удаление строк с пропущенными значениями':
            self.settings['fillna'] = 'dropna'
        else:
            self.settings['fillna'] = 'median'
        with open('settings.py', 'rb') as f:
            data = pickle.load(f)
        data['MODULE_SETTINGS']['data'].update(self.settings)
        with open('settings.py', 'wb') as f:
            pickle.dump(data, f)
        self.hide()
        self.child.show()
