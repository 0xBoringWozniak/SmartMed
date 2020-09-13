from GUI import GUI
from backend import ModuleManipulator


if __name__ == '__main__':
	gui = GUI()
	settings = gui.start_gui()  # empty

<<<<<<< HEAD
    # temp settings
    settings = {'MODULE': 'STATS',
                'MODULE_SETTINGS': {
                    'data': {'preprocessing': {'AUTO': False, 'fillna': 'mean'},
                             'path': 'C:/projects/SmartMed/backend/modules/random.csv'},
                    'metrics': {
                        'mean': True,
                        'std': True,
                        'max': True
                    }
                }
                }
=======
	# temp settings
	settings = {'MODULE': 'STATS',
				'MODULE_SETTINGS': {
					'AUTO': False,
					'data': {'preprocessing': {
											   'AUTO': False,
											   'fillna': 'mean',
											   'encoding': 'label_encoding',
											   'scaling': False
											  },
							'path': 'winequality-white.csv'
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
>>>>>>> dev

	ModuleManipulator(settings).start()
