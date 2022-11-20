import pandas as pd
import numpy as np
from src.config import logger 
import logging
import math
import warnings
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)

class Transformer:
    def __init__(self,logger = logging.getLogger()):
        self._logger = logger
        self._input_df = None
        self._output_df = None

    def _get_logger(self):
        return self._logger
    
    def _get_input_data(self):
        return self._input_df
    
    def _get_output_data(self):
        return self._output_df

    def _set_raw_data(self,input_df):
        self._input_df = input_df

class NumericalVariableCleaner(Transformer):
    """
    Date: 2022-11-18
    Description: The goal of this class is to clean the dataframe by
    stripping the unit of measure from dataframe value --> i.e. 100mg to 100
    """
    def __init__(self, logger = logging.getLogger()):
        super().__init__(logger=logger)
        self._logger = logger

    def clean(self,input_df,col_keep = None,col_drop = None):
        self._logger.info('Cleaning numerical fields.')
        super()._set_raw_data(input_df)
        self._logger.info('Stored data frame inside object.')
        self._logger.info('Selecting columns to clean.')
        if col_keep == None:
            self._logger.info('Columns to clean is not specified.')
            if col_drop == None:
                self._logger.info('All columns will be cleaned.')
                col_keep = input_df.columns.values
            else:
                self._logger.info(f'Dropping the following specified columns: {col_drop}')
                col_keep = np.delete(input_df.columns.values, np.where(input_df.columns.values == col_drop))
        self._logger.info(f'Cleaning the following fields: {col_keep}')
        output_df = input_df[col_drop]
        measure_dict = {}
        self._logger.info('Removing unit of measure from the actual value in the columns such that the fields are purely numerical.')
        self._logger.info('Storing unit of measure in the _measure_units attribute of the object.')
        for col in col_keep:
            if col == col_drop:
                continue
            try:
                output_df[col] = pd.to_numeric(input_df[col].str.extract('([-+]?\d*\.?\d+)')[0],errors='coerce')
                #output_df[f'{col}_measure'] = output_df[col].str.extract('(\D+)')
                measure_col_list = list(set((input_df[col].str.extract(r"([a-z]+)").values).ravel()))
                measure_col_list = [x for x in measure_col_list if str(x) != 'nan']
                measure_dict.update({col:measure_col_list})
            except:
                continue
        self._col_keep = col_keep
        self._output_df = output_df
        self._measure_units = measure_dict
        self._logger.info('Finished cleaning.')
        return self._output_df


class NumericalMasker(Transformer):
    """
    Date: 2022-11-18
    Description: The goal of this class is to clean the dataframe by
    stripping the unit of measure from dataframe value --> i.e. 100mg to 100
    """
    def __init__(self):
        super().__init__(logger=logger)
        self._logger = logger
        self._corr_matrix = None

    def _set_corr_matrix(self,input_df):
        self._corr_matrix = input_df.corr().abs()
        return self._corr_matrix

    def _set_ranking_matrix(self,input_df,rank_method):
        self._ranking_matrix = input_df.rank(ascending=False, method=rank_method)
        return self._ranking_matrix 

    def _select_mask_indices(self, input_df, target_col, selected_cols, col_weights):


    def mask(self, input_df, target_col, rank_method = 'first', normalize_weights = True, selected_cols = None,seed = None,frac_na=0.1, max_corr = 0.9, col_weights = None, no_cols = None):
        super()._set_raw_data(input_df)
        corr_matrix = self._set_corr_matrix(input_df)
        if selected_cols == None:
            if no_cols == None:
                tot_no_cols = len(input_df.columns.values)
                no_cols = math.sqrt(tot_no_cols)
        params = dict(corr_matrix[target_col].nlargest(no_cols))
        ranking_matrix = self._set_ranking_matrix(input_df,rank_method)
        # weighted_ranking_matrix = ranking_matrix.assign(**params).mul(ranking_matrix).sum(1)
        weighted_ranking_matrix = ranking_matrix.mul(pd.Series(params), axis=1)
        weighted_index = weighted_ranking_matrix.rank().sum(axis=1)
        test_weights = [0.2, 0.2, 0.2, 0.4]
        df1.sample(n = 3, weights = test_weights)
        
        if normalize_weights == True:
            #
            corr_matrix[target_col]
            print('placeholder')
        input_df['order'] = input_df[[selected_cols]].rank().sum(axis=1)
        input_df.sort_values('order', inplace=True, ascending=False) 
            
        # weighted random sampling
        self._logger.info(f"Processing df with shape '{input_df.shape}'.")
        self._set_corr_matrix(input_df)

