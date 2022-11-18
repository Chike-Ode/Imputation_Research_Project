import pandas as pd
import numpy as np
from src.config import logger 
import logging

logging.basicConfig(level=logging.INFO)

class Transformer:
    def __init__(self,logger = logging.getLogger()):
        self._logger = logger
        self._X = None

    def _get_logger(self):
        return self._logger
    
    def _get_raw_data(self):
        return self._X

    def _set_raw_data(self,X):
        self._X = X

class NumericalVariableCleaner(Transformer):
    """
    Date: 2022-11-18
    Description: The goal of this class is to clean the dataframe by
    stripping the unit of measure from dataframe value --> i.e. 100mg to 100
    """
    def __init__(self,frac_na=0.2, corr_str = 0, type="Missing at Random"):
        self._logger = logger
        self._X = None
        self._frac_na = frac_na
        self._corr_matrix = None

    def clean(self,X,col_keep = None,col_drop = None):
        super()._set_raw_data(X)
        if col_keep != None:
            pass
        else:
            if col_drop == None:
                col_keep = X.columns.values
            else:
                col_keep = np.delete(X.columns.values, np.where(X.columns.values == col_drop))
        output_df = X[col_drop]
        measure_dict = {}
        for col in col_keep:
            if col == col_drop:
                continue
            try:
                output_df[col] = X[col].str.extract('([-+]?\d*\.?\d+)')
                #output_df[f'{col}_measure'] = output_df[col].str.extract('(\D+)')
                measure_col_list = list(set((X[col].str.extract(r"([a-z]+)").values).ravel()))
                measure_col_list = [x for x in measure_col_list if str(x) != 'nan']
                measure_dict.update({col:measure_col_list})
            except:
                continue
        self._col_keep = col_keep
        self._output_df = output_df
        self._measure_units = measure_dict
        return self._output_df


class Masker(Transformer):
    """
    Date: 2022-11-18
    Description: The goal of this class is to clean the dataframe by
    stripping the unit of measure from dataframe value --> i.e. 100mg to 100
    """
    def __init__(self,frac_na=0.2, corr_str = 0, type="Missing at Random"):
        self._logger = logger
        self._X = None
        self._frac_na = frac_na
        self._corr_matrix = None

    def _set_corr_matrix(self,X):
        self._corr_matrix = X.corr().abs()

    def fit(self,X,frac,column_names):
        super()._set_raw_data(X)
        # weighted random sampling
        self._X = X
        self._logger.info(f"Processing df with shape '{X.shape}'.")
        self._set_corr_matrix(X)

