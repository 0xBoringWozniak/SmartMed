import numpy as np
from sklearn.linear_model import LinearRegression

from .ModelInterface import Model


class LinearRegressionModel(Model):
    
    def __init__(self, x, y):
        self.model = LinearRegression()
        super().__init__(x, y)
        
    def score(self) -> float:
        return self.model.score(self.x, self.y)
    
    def get_resid(self) -> np.array:
        return self.model.coef_