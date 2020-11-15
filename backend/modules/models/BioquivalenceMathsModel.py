import numpy as np

from sklearn.metrics import auc


class BioquivalenceMathsModel:

    def get_auc(x: np.array, y: np.array) -> float:
        return auc(x, y)

    def get_log_array(x: np.array) -> np.array:
        return np.log(x)
