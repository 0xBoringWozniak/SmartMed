import unittest

from backend.modules.dash.Dashboard import Dashboard
from backend.DashExceptions import ModelChoiceException

class TestDashboard(unittest.TestCase):
    def setUp(self):
        self.dashboard = Dashboard()

    def test_start(self):
        self.dashboard.start()

    def test_for_linear(self):
        self.dashboard.setting = {"model": "linear"}
        self.dashboard.start()

    def test_for_bad_model(self):
        self.dashboard.setting = {"model": "bad_model_name"}
        with self.assertRaises(ModelChoiceException):
            self.dashboard.start()
