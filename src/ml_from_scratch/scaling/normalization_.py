from typing import Union

import numpy as np
import pandas as pd

from ml_from_scratch.transformation import Transformer



class Normalization(Transformer):
    maxs = None
    mins = None

    def _fit(self, X: pd.DataFrame, y: Union[pd.DataFrame, pd.Series] = None) -> 'Normalization':
        self.maxs = np.amax(X, axis=0)
        self.mins = np.amin(X, axis=0)
        
        return self

    def _transform(self, x: pd.DataFrame, y: Union[pd.DataFrame, pd.Series] = None) -> pd.DataFrame:
        return (x - self.mins) / (self.maxs - self.mins)
        

    def inverse_transform(self, X: pd.DataFrame, y: Union[pd.DataFrame, pd.Series] = None) -> pd.DataFrame:
        return X * (self.maxs - self.mins) + self.mins



if __name__ == '__main__':
    # Example usage
    X = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [1, 2, 3, 4, 5], 'c': [1, 2, 3, 4, 5]})
    print(X)
    scaler = Normalization()
    scaler.fit(X)
    print(scaler.maxs, scaler.mins)
    print(scaler.transform(X))
    print(scaler.inverse_transform(scaler.transform(X)))