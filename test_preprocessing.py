import unittest

import pandas as pd

class TestPreprocessing(unittest.TestCase):
    def SetUp(self):
        self.prepocessor = PandasPreprocessor({})
        self.df = pd.DataFrame({"col_str": ["A", None, "C"], "col_float": [1.0, 2.0, 3.0]})

    def set_df(self):
        self.prepocessor.df = self.df.copy()

    def test_fillna():
        self.set_df()
        self.prepocessor.__fillna("fill_str")
        self.assertEqual(self.prepocessor.df.isna().sum() == 0)

        self.set_df()
        self.prepocessor.__fillna("mean")
        self.assertEqual(self.prepocessor.df.isna().sum() == 0)

        self.set_df()
        self.prepocessor.__fillna("median")
        self.assertEqual(self.prepocessor.df.isna().sum() == 0)
