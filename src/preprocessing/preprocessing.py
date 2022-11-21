import pandas as pd
import numpy as np
from src.config import logger 
import logging
import math
import warnings
from sklearn.preprocessing import StandardScaler
import random
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

    def _set_ranking_matrix(self,input_df,rank_method, na_option = 'last'):
        self._ranking_matrix = input_df.rank(ascending = False, method = rank_method, na_option = 'first')
        return self._ranking_matrix 

    def _select_mask_indices(self, input_df, target_col, selected_cols, col_weights):
        pass

    def _create_index_weights(self):
        

    def mask(self, input_df, target_col, k = 4, n = None, prob_range_non_mask = (0,0.5), prob_range_mask = (0.5,1), frac_na=0.1, rank_method = 'first', normalize_weights = True, selected_cols = None, seed = None,  max_corr = 0.9, min_corr = 0.3, col_weights = None, no_cols = None):
        super()._set_raw_data(input_df)
        if selected_cols == None:
            if no_cols == None:
                tot_no_cols = len(input_df.select_dtypes(include=np.number).columns.values)
                no_cols = int(math.sqrt(tot_no_cols))
        else:
            input_df = input_df[selected_cols]
        corr_matrix = self._set_corr_matrix(input_df)
        if col_weights == None:
            col_weights = dict(corr_matrix[target_col].nlargest(no_cols))
        col_keep = col_weights.keys()
        ranking_matrix = self._set_ranking_matrix(input_df[col_keep],rank_method)
        # weighted_ranking_matrix = ranking_matrix.assign(**params).mul(ranking_matrix).sum(1)
        weighted_ranking_matrix = ranking_matrix.mul(pd.Series(col_weights), axis=1)
        #weighted_index = weighted_ranking_matrix.sum(axis=1).sort_values().reset_index()
        similar_ordered_index = weighted_ranking_matrix.sum(axis=1).sort_values().rename_axis('original_index').reset_index(name='score')
        if n == None:
            n = int(frac_na *input_df.shape[0])
        cluster_df_list = []
        end = 0
        increment = int(similar_ordered_index.shape[0]/k)
        for i in range(1,k + 1):
            start = end
            end = start + increment
            # start = int((similar_ordered_index.shape[0]/(k +1)) * i)
            # end = int((similar_ordered_index.shape[0]/(k + 1)) * (i + 1))
            nrows = range(start,end)
            k_n = int(n/k)
            ix = random.randint(nrows.start, nrows.stop-k_n)
            cluster_index_df = similar_ordered_index.iloc[ix:ix+k_n, :]
            cluster_index_df['cluster_id'] = i
            cluster_df_list.append(cluster_index_df)
        all_cluster_index_df = pd.concat(cluster_df_list)
        weights_all = np.random.uniform(low=prob_range_non_mask[0], high=prob_range_non_mask[1], size=(input_df.shape[0],)) # Non masked probability distribution
        update_index_list = all_cluster_index_df['original_index'].to_list()
        weights_all[update_index_list] = np.random.uniform(low=prob_range_mask[0], high=prob_range_mask[1], size=(all_cluster_index_df.shape[0],)) # Masking probability distribution
        mask_df = input_df.sample(n = n, weights = weights_all,random_state = seed) # weighted random sample of rows to mask
        mask_df['mask_ind'] = 1
        output_df = input_df.join(mask_df[['mask_ind']]) # join to add mask_ind column to raw data
        output_df['mask_ind'] = output_df['mask_ind'].notnull().astype(int) # replacing null in mask_ind field from join with 0 
        if normalize_weights == True:
            #
            corr_matrix[target_col]
            print('placeholder')
        input_df['order'] = input_df[[selected_cols]].rank().sum(axis=1)
        input_df.sort_values('order', inplace=True, ascending=False) 
            
        # weighted random sampling
        self._logger.info(f"Processing df with shape '{input_df.shape}'.")
        self._set_corr_matrix(input_df)

