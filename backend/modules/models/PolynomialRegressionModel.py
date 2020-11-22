import numpy as np
from sklearn.linear_model import LogisticRegression

from .ModelInterface import Model


class PolynomialRegressionModel(Model):
    
    def __init__(self, x, y): 
        poly = PolynomialFeatures(degree=2)
        x = poly.fit_transform(self.x)
        y = poly.fit_transform(self.y)
        
        self.model = LinearRegression()
        super.__init__(x, y)