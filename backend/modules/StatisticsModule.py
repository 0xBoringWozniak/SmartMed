import pandas as pd

from .ModuleInterface import Module
from .dash import StatisticsDashboard


class StatisticsModule(Module, StatisticsDashboard):

	def _prepare_data(self):
		self.pp.preprocess()
		return self.pp.df

	def _prepare_dashboard_settings(self):
		settings = dict()

		settings['metrics'] = []
		for metric in self.settings['metrics'].keys():
			if self.settings['metrics'][metric]:
				settings['metrics'].append(metric)

		settings['graphs'] = []
		for graph in self.settings['graphs'].keys():
			if self.settings['graphs'][graph]:
				settings['graphs'].append(graph)

		# replace log and linear to linlog multiple graph
		if 'linear' in settings['graphs'] and 'log' in settings['graphs']:
			settings['graphs'].append('linlog')
			settings['graphs'].remove('linear')
			settings['graphs'].remove('log')

		if 'box' in settings['graphs'] and 'hist' in settings['graphs']:
			settings['graphs'].append('boxhist')

		self.graph_to_method = {
			'linear': self._generate_linear,
			'log': self._generate_log,
			'corr': self._generate_corr,
			'heatmap': self._generate_heatmap,
			'scatter': self._generate_scatter,
			'hist': self._generate_hist,
			'box': self._generate_box,
			'linlog': self._generate_linlog,
			'dotplot': self._generate_dotplot,
			'piechart': self._generate_piechart,
			'boxhist': self._generate_box_hist
		}

		settings['data'] = self.data

		return settings

	def _prepare_dashboard(self):
		pass
