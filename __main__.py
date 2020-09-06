from GUI import GUI
from backend import ModuleManipulator


if __name__ == '__main__':
    gui = GUI()
    settings = gui.start_gui()  # empty

    # temp settings
    settings = {'MODULE': 'STATS',
                'MODULE_SETTINGS': {
                    'data': {'preprocessing': {'AUTO': False, 'fillna': 'mean'},
                             'path': '/Users/ba/Documents/SmartMed/backend/modules/fr_bitmex.csv'},
                    'metrics': {
                        'mean': True,
                        'std': True,
                        'max': True
                    }
                }
                }

    ModuleManipulator(settings).start()
