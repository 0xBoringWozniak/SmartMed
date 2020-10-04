from abc import ABC, abstractmethod

import dash

# logging decorator
import sys
sys.path.append("..")
from logs.logger import debug


class Dashboard(ABC):
	'''Dashboard Interface'''
	def __init__(self):
		external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
		self.app = dash.Dash(server=True, external_stylesheets=external_stylesheets)

	@debug
	@abstractmethod
	def _generate_layout(self):
		raise NotImplementedError

	@debug
	def start(self, debug=False):
		self.app.layout = self._generate_layout()
		self.app.run_server(debug=debug)
