import pickle

from .dash.PredictionDashboard import PredictionDashboard


class DashboardModelChooseExcpetion(Exception):
	pass


class DahsboardSwitcher():
    def choose(self):
    	with open('settings.py', 'rb') as f:
    		model_type = pickle.load(f)['MODULE_SETTINGS']['model']

    	if model_type == 'linreg':
    		return PredictionDashboard

    	else:
    		raise DashboardModelChooseExcpetion

