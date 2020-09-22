import pandas as pd

from .ModuleInterface import Module
from .dash import StatisticsDashboard
from .dataprep import PandasPreprocessor


class StatisticsModule(Module, StatisticsDashboard):
	def _prepare_data(self):
		pp = PandasPreprocessor(self.settings['data'])
		pp.preprocess()
		return pp.df
	
	def _prepare_dashboard_settings(self):
		settings = dict()

		settings['metrics'] = []
		for metric in self.settings['metrics'].keys():
			if self.settings['metrics']:
				settings['metrics'].append(metric)

		settings['data'] = self.data

		return settings

	def _prepare_dashboard(self):
		pass
