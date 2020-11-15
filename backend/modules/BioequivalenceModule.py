from .ModuleInterface import Module
from .dash import BioequivalenceDashboard
from .models import BioquivalenceMathsModel


class BioequivalenceModule(Module, BioequivalenceDashboard, BioquivalenceMathsModel):

	def _prepare_data(self):
		self.pp.preprocess()
		return self.pp.df

	def _prepare_dashboard_settings(self):
		pass

	def _prepare_dashboard(self):
		pass