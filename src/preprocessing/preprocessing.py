import pandas as pd
import numpy as np
from src.config import logger 
import logging
import math
import warnings
# from sklearn.preprocessing import StandardScaler
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

    def clean(self,input_df,col_keep = None,col_ignore = None):
        self._logger.info('Cleaning numerical fields.')
        super()._set_raw_data(input_df)
        self._logger.info('Stored data frame inside object.')
        self._logger.info('Selecting columns to clean.')
        if col_keep == None:
            self._logger.info('Columns to clean is not specified.')
            if col_ignore == None:
                self._logger.info('All columns will be cleaned.')
                col_keep = input_df.columns.values
            else:
                self._logger.info(f'Dropping the following specified columns: {col_ignore}')
                col_keep = np.delete(input_df.columns.values, np.where(input_df.columns.values == col_ignore))
        self._logger.info(f'Cleaning the following fields: {col_keep}')
        output_df = input_df[col_ignore]
        measure_dict = {}
        self._logger.info('Removing unit of measure from the actual value in the columns such that the fields are purely numerical.')
        self._logger.info('Storing unit of measure in the _measure_units attribute of the object.')
        for col in col_keep:
            if col == col_ignore:
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
    Description: The goal of this class is to mask a certain field of the data set
    By looking at the most correlated attributes to that feature
    """
    def __init__(self):
        super().__init__(logger=logger)
        self._logger = logger
        self._corr_matrix = None

    def _set_corr_matrix(self,input_df):
        self._corr_matrix = input_df.corr().abs()
        self._logger.info('Finished creating correlation matrix.')
        return self._corr_matrix

    def _set_ranking_matrix(self,input_df,rank_method, pct_rank, na_option = 'bottom'): # TODO fix na_option
        self._ranking_matrix = input_df.rank(ascending = False, method = rank_method, na_option = na_option, pct = pct_rank)
        self._logger.info('Finished creating ranking matrix.')
        return self._ranking_matrix 

    def _create_index_weights(self, all_cluster_index_df, input_df, prob_range_non_mask, prob_range_mask):
        weights_all = np.random.uniform(low=prob_range_non_mask[0], high=prob_range_non_mask[1], size=(input_df.shape[0],)) # Non masked probability distribution
        update_index_list = all_cluster_index_df['original_index'].to_list()
        weights_all[update_index_list] = np.random.uniform(low=prob_range_mask[0], high=prob_range_mask[1], size=(all_cluster_index_df.shape[0],)) # Masking probability distribution
        self._weights_all = weights_all
        self._logger.info('Finished creating index weights.')
        return weights_all

    def _create_output_df(self,input_df,weights_all,seed,n,target_col):
        mask_df = input_df.sample(n = n, weights = weights_all,random_state = seed) # weighted random sample of rows to mask
        mask_df[f'mask_ind_{target_col}'] = 1
        output_df = input_df.join(mask_df[[f'mask_ind_{target_col}']]) # join to add mask_ind column to raw data
        output_df[f'mask_ind_{target_col}'] = output_df[f'mask_ind_{target_col}'].notnull().astype(int) # replacing null in mask_ind field from join with 0 
        self._output_df = output_df
        return output_df

    def _create_clusters(self,similar_ordered_index,k,n):
        cluster_df_list = []
        end = 0
        increment = int(similar_ordered_index.shape[0]/k)
        for i in range(1,k + 1):
            start = end
            end = start + increment
            nrows = range(start,end)
            k_n = int(n/k)
            ix = random.randint(nrows.start, nrows.stop-k_n)
            cluster_index_df = similar_ordered_index.iloc[ix:ix+k_n, :]
            cluster_index_df['cluster_id'] = i
            cluster_df_list.append(cluster_index_df)
        
        all_cluster_index_df = pd.concat(cluster_df_list)
        self._all_cluster_index_df = all_cluster_index_df
        self._logger.info('Finished creating clusters in the dataframe.')
        return all_cluster_index_df

    def _create_similar_ordered_index(self,ranking_matrix,col_weights):
        weighted_ranking_matrix = ranking_matrix.mul(pd.Series(col_weights), axis=1)
        similar_ordered_index = weighted_ranking_matrix.sum(axis=1).sort_values().rename_axis('original_index').reset_index(name='score')
        self._similar_ordered_index = similar_ordered_index
        self._logger.info('Finished ordering similar rows.')
        return similar_ordered_index
        
    def mask(self, input_df, target_col, k = 4, n = None, no_cols_frac = 0.5, no_cols = None, prob_range_non_mask = (0,0.5), prob_range_mask = (0.5,1), frac_na=0.1, rank_method = 'first', normalize_weights = True, selected_cols = None, seed = None,  max_corr = 0.9, min_corr = 0.3, col_weights = None, pct_rank = True):
        super()._set_raw_data(input_df)
        self._logger.info(f'Starting masking process for the {target_col} field.')
        self._logger.info('Selecting key columns.')
        if n == None:
            n = int(frac_na * input_df.shape[0]) # number of rows to mask
        if selected_cols == None:
            if no_cols == None:
                tot_no_cols = len(input_df.select_dtypes(include=np.number).columns.values)
                no_cols = int(tot_no_cols * no_cols_frac) # number of columns to be considered
        else:
            input_df = input_df[selected_cols]
        self._logger.info(f'Masking {n} rows which make up for {"{:.2f}".format(float(n/input_df.shape[0])*100)}% of the observations into {k} clusters.')
        corr_matrix = self._set_corr_matrix(input_df)
        target_correlations = corr_matrix[target_col].nlargest(no_cols)
        if normalize_weights == True: # TODO fix np.where issue
            # adjusting minimum and maximum weights to be applied to ranked fields
            self._logger.info('Normalizing weights.')
            target_correlations[target_correlations > max_corr] = max_corr
            target_correlations[target_correlations < min_corr] = min_corr
            # target_correlations = target_correlations.values.tolist()
            # target_correlations = np.where(target_correlations>max_corr,max_corr,np.where(target_correlations<min_corr,min_corr,target_correlations))
        if col_weights == None:
            col_weights = dict(target_correlations)
        col_keep = col_weights.keys()
        ranking_matrix = self._set_ranking_matrix(input_df[col_keep],rank_method, pct_rank)
        similar_ordered_index = self._create_similar_ordered_index(ranking_matrix,col_weights)
        all_cluster_index_df = self._create_clusters(similar_ordered_index,k,n)
        weights_all = self._create_index_weights(all_cluster_index_df, input_df, prob_range_non_mask, prob_range_mask)
        output_df = self._create_output_df(input_df,weights_all,seed,n,target_col)
        return output_df
        
