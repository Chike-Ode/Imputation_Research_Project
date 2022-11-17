import pandas as pd
import numpy as np
from src.config import logger 


class Transformer():
    def __init__(self,frac_na=0.2, corr_str = 0, type="Missing at Random"):
        self._logger = logger
        self._X = None
        self._frac_na = frac_na
        self._corr_matrix = None

    def _get_logger(self):
        return self._logger
    
    def _get_raw_data(self):
        return self._X

    def _set_corr_matrix(self,X):
        self._corr_matrix = X.corr().abs()

    def masker(self,X):
        pass

    def int_col_converter(self,X):
        output_df = X[['name']]
        measure_dict = {}
        for col in X.columns.values:
            try:
                #output_df[f'{col}_measure'] = output_df[col].str.extract('(\D+)')
                measure_col_list = list(set((X[col].str.extract(r"([a-z]+)").values).ravel()))
                measure_dict.update({col:measure_col_list})
                output_df[f'{col}_val'] = X[col].str.extract(r'(\d+.\d+)')
            except:
                continue
        self._output_df = output_df
        self._measure_dict = measure_dict
        return self._output_df

    def fit(self,X,frac,column_names):
        self._X = X
        self._logger.info(f"Processing df with shape '{X.shape}'.")
        self._set_corr_matrix(X)

