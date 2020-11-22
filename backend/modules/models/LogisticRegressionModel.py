import numpy as np
from sklearn.linear_model import LogisticRegression

from .ModelInterface import Model


class LogisticRegressionModel(Model):
    
    def __init__(self, x, y):
        self.model = LogisticRegression()
        super().__init__(x, y)
