from abc import ABC, abstractmethod
from typing import Dict


class Module(ABC):
	def __init__(self, settings: Dict):
		self.settings = settings
		super().__init__()

	@abstractmethod
	def _prepare_data(self):
		'''manipulate data according to settings'''
		raise NotImplementedError
	
	@abstractmethod
	def _prepare_dashboard_settings(self):
		'''construct dashboard settings'''
		raise NotImplementedError

	@abstractmethod
	def _prepare_dashboard(self):
		'''generate dash using DashContsructor'''
		raise NotImplementedError

	def run(self):
		'''
		start module
		1. take settings
		2. manipulate data (prepare data, build models and calculate metrics)
		3. prepare dash
		4. start dash
		'''
		self.data = self._prepare_data()
		self.settings = self._prepare_dashboard_settings()
		self._prepare_dashboard()
		self.start()


		
