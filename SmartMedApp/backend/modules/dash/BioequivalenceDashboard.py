import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd

from .Dashboard import Dashboard


class BioequivalenceDashboard(Dashboard):
	
	def _generate_layout(self):
		return html.Div([])