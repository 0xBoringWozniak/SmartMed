from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from .BaseModel import BaseModel


class PolynomialRegressionModel(BaseModel):

    def __init__(self, x, y, degree=2, col_idx=1):
        poly = PolynomialFeatures(degree)
        x_poly = poly.fit_transform(x)[:, 1 + col_idx]
        super().__init__(LinearRegression, x_poly, y)
