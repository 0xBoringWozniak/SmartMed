from typing import Dict

from .ModuleInterface import Module
from .dash import PredictionDashboard


class PredictionModule(Module, PredictionDashboard):
	pass
