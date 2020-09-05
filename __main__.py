from GUI import GUI
from backend import ModuleManipulator


gui = GUI()
settings = gui.display()

settings = {'MODULE': 'STATS',
			'MODULE_SETTINGS': {
								'data': {'path': '/Users/ba/Documents/SmartMed/backend/modules/fr_bitmex.csv'},
								'metrics': {
											'mean': True,
											'std': True,
											'max': True
											}
								}
			}


module = ModuleManipulator(settings)
module.start()