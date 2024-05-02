from typing import Union
from typing import Callable

import pandas as pd
import numpy as np

from ml_from_scratch.statistics import f_classif
from ml_from_scratch.transformation import Transformer



class SelectPercentile(Transformer):
    def __init__(self, k : int, score_func: Callable = f_classif):
        self.k = k
        self.score_func = score_func
        self.score = None
        self.percentiles = None
        self.selected_features = None

    def _fit(self, X: pd.DataFrame, y: Union[pd.DataFrame, pd.Series] = None) -> 'SelectPercentile':
        self.scores = self.score_func(X, y)
        self.percentiles = np.percentile(self.scores, np.linspace(0, 100, num = X.shape[1]+1))
        cutoff_percentile = np.percentile(self.scores, (100 - self.k))

        selected_features = np.where(self.scores >= cutoff_percentile)[0]
        self.selected_features = selected_features.tolist()

        return self

    def _transform(self, X: pd.DataFrame, y: Union[pd.DataFrame, pd.Series] = None) -> pd.DataFrame:
        return X.loc[:, self.selected_features]




if __name__ == '__main__':
    # Example usage
    X = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [1, 2, 3, 4, 5], 'c': [1, 1, 1, 1, 1]})
    y = pd.Series([0, 1, 0, 1, 0])
    selector = SelectPercentile(k=4)
    selector.fit(X, y)
    print(selector.percentiles)
    X = selector.transform(X)
    print(X)