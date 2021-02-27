from typing import Dict

from .modules import *


class ModuleChoiceException(Exception):
    pass


class ModuleManipulator:

    def __init__(self, settings: Dict):
        self.settings = settings

    def start(self):
        if self.settings['MODULE'] == 'STATS':
            module = StatisticsModule(self.settings['MODULE_SETTINGS'])
            print('StatisticsModule with settings:{}'.format(
                self.settings['MODULE_SETTINGS']))
        elif self.settings['MODULE'] == 'PREDICT':
            print(self.settings['MODULE_SETTINGS'])
            module = PredictionModule(self.settings['MODULE_SETTINGS'])
            print('PredictionModule with settings: {}'.format(
                self.settings['MODULE_SETTINGS']))
        elif self.settings['MODULE'] == 'BIOEQ':
            module = BioequivalenceModule(self.settings['MODULE_SETTINGS'])
            print('BioequivalenceModule with settings: {}'.format(
                self.settings['MODULE_SETTINGS']))
        else:
            raise ModuleChoiceException

        module.run()
