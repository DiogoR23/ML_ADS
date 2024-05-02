from typing import Union

import numpy as np
import pandas as pd

from ml_from_scratch.transformation import Transformer



class ModeImputer(Transformer):
    modes = None

    def _fit(self, X: pd.DataFrame, y: Union[pd.DataFrame, pd.Series] = None) -> 'ModeImputer':
        self.modes = X.mode()

        return self

    def _transform(self, X: pd.DataFrame, y: Union[pd.DataFrame, pd.Series] = None) -> pd.DataFrame:
        for col in X.columns:
            X[col].fillna(self.modes[col].values[0], inplace = True)

        return X



if __name__ == "__main__":
    X = pd.DataFrame({'a': [np.nan, 22, 3, 9, 9], 'b': [10, np.nan, 3, 4, 4], 'c': [5, 29, np.nan, 4, 5]})
    print(X)
    imputer = ModeImputer()
    imputer.fit(X)
    print(f"Modes:\n{imputer.modes}")
    print(imputer.transform(X))