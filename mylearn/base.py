import numpy as np


class BaseEstimator:
    def _validate_data(
            self,
            X="no validation",
            copy=False,
            order=None,
            dtype="numeric"
    ):
        return X