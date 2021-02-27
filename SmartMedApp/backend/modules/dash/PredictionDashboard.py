import pylatex
import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from .markdown_text import *

from .DashExceptions import ModelChoiceException
from .Dashboard import Dashboard


class PredictionDashboard(Dashboard):
    def _generate_layout(self):
        if self.setting['model'] == 'linear':
            LinearRegressionDashboar()._generate_layout()
        else:
            raise ModelChoiceException

class LinearRegressionDashboar(Dashboard):
    def _generate_layout(self):
        print('LinearRegressionDashboar test') 
        return html.Div(graph_list)
