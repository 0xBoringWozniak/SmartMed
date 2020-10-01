from GUI import GUI
from backend import ModuleManipulator
import pickle


if __name__ == '__main__':
	gui = GUI()
	gui.start_gui()  # empty

	with open('settings.py', 'rb') as f:
		settings = pickle.load(f)
		print(settings)

	
	# temp settings
	'''settings = {'MODULE': 'STATS',
				'MODULE_SETTINGS': {
					'AUTO': False,
					'data': {'preprocessing': {
											   'AUTO': False,
											   'fillna': 'mean',
											   'encoding': 'label_encoding',
											   'scaling': False
											  },
							'path': 'titanic.csv'
							},
					'metrics': {
						'AUTO': False,
						'count': True,
						'mean': True,
						'std': True,
						'max': True,
						'min': True,
						'25%': True,
						'50%': True,
						'75%': True
					},
					'grahics': {
						'AUTO': False,
						'linear': True,
						'log': True,
						'corr': True,
						'heatmap': True,
						'scatter': True,
						'hist': True,
						'box': True
					}
				}
				}
				'''


	ModuleManipulator(settings).start()
