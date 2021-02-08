from abc import ABC, abstractmethod

import time
import sys

import dash
import webbrowser

from SmartMedApp.logs.logger import debug


class Dashboard(ABC):
	'''

	Dashboard Interface

	Each ConcreteDashboar inreases port number 
	and Dashboar_i is opened on localhost with port = 8000 + i
	in daemon thread

	'''
	port = 8000
	def __init__(self):

		# include general styleshits and scripts
		external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
		external_scripts = ['https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-MML-AM_CHTML']

		# create Dash(Flask) server
		self.app = dash.Dash(
			server=True, external_stylesheets=external_stylesheets, external_scripts=external_scripts)

		# increase port
		Dashboard.port += 1

	@debug
	@abstractmethod
	def _generate_layout(self):
		'''
		abstractmethod to generate dashboard layout
		'''
		raise NotImplementedError

	@debug
	def start(self, debug=False):
		# generate layout
		self.app.layout = self._generate_layout()

		# set port
		port = Dashboard.port

		# open dashboard
		webbrowser.open(f"http://127.0.0.1:{port}/")

		# run dashboard
		self.app.run_server(debug=debug, port=port)
