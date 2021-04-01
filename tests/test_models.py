import unittest

from backend.modules.models.LinearRegressionModel import LinearRegressionModel
from backend.modules.models.LogisticRegressionModel import LogisticRegressionModel
from backend.modules.models.PolynomialRegressionModel import PolynomialRegressionModel


class TestModel(unittest.Testcase):
    def setUp(self):
        self.linreg = LinearRegressionModel()
        self.logreg = LogisticRegressionModel()
        self.polyreg = PolynomialRegressionModel()

    def test_score(self):
        self.linreg.score()
        self.logreg.score()
        self.polyreg.score()

    def test_get_resid(self):
        self.linreg.get_resid()
        self.logreg.get_resid()
        self.polyreg.get_resid()
