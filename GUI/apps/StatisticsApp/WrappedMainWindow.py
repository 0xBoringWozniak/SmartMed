import pickle

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import (QWidget, QToolTip, QPushButton, QApplication)
from .MainWindow import MainWindow


class WrappedMainWindow(MainWindow, QtWidgets.QMainWindow):
	def __init__(self):
		super().__init__()
		self.setupUi(self)
		self.settings = { 'MODULE_SETTINGS':{}, 'MODULE' : 'STATS'}
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
		self.commandLinkButton.clicked.connect(self.path_to_file)

	def next(self):
		value1 = self.comboBox1.currentText()
		value2 = self.comboBox2.currentText()
		if value1 == 'средним/модой (аналогично)':
			self.settings['MODULE_SETTINGS']['data']['fillna'] = 'mean'
		elif value1 == 'заданным значием (требуется ввод для каждого столбца отдельно)':
			self.settings['MODULE_SETTINGS']['data']['fillna'] = 'exact_value'
		elif value1 == 'откидывание строк с пропущенными значениями':
			self.settings['MODULE_SETTINGS']['data']['fillna'] = 'dropna'
		else:
			self.settings['MODULE_SETTINGS']['data']['fillna'] = 'median'
		if value2 == 'label encoding':
			self.settings['MODULE_SETTINGS']['data']['encoding'] = 'label_encoding'
		elif value2 == 'dummy encoding':
			self.settings['MODULE_SETTINGS']['data']['encoding'] = 'dummy encoding'
		elif value2 == 'one hot encoding':
			self.settings['MODULE_SETTINGS']['data']['encoding'] = 'one hot encoding'
		else:
			self.settings['MODULE_SETTINGS']['data']['encoding'] = 'binary encoding'
		with open('settings.py', 'wb') as f:
			pickle.dump(self.settings, f)
		self.hide()
		self.leaf_2.show()


	def path_to_file(self):
		settings = {}
		settings['data'] = {'preprocessing': {
											   'AUTO': False,
											   'fillna': 'mean',
											   'encoding': 'label_encoding',
											   'scaling': False
											  },
								'path': ''
								}
		settings['data']['path'] = QtWidgets.QFileDialog.getOpenFileName()[0]
		self.settings['MODULE_SETTINGS'].update(settings)