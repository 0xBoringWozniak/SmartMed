from typing import Dict

import logging

from .modules import *

logging.basicConfig(filename='~/../logs/start.log', level=logging.DEBUG)


class ModuleManipulator:
    def __init__(self, settings: Dict):
        self.settings = settings
    
    def start(self):
        if self.settings['MODULE'] == 'STATS':
            module = StatisticsModule(self.settings['MODULE_SETTINGS'])
            logging.debug('StatisticsModule was created. Settings:\n{}'.format(self.settings['MODULE_SETTINGS']))
        elif self.settings['MODULE'] == 'PREDICT':
            module = PredictionModule(self.settings['MODULE_SETTINGS'])
            logging.debug('PredictionModule was created. Settings:\n{}'.format(self.settings['MODULE_SETTINGS']))
        elif  self.settings['MODULE'] == 'BIOEQ':
            module = BioequivalenceModule(self.settings['MODULE_SETTINGS'])
            logging.debug('BioequivalenceModule was created. Settings:\n{}'.format(self.settings['MODULE_SETTINGS']))

        module.run()