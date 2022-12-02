import pandas as pd
import numpy as np


def get_variation_coefficients(df,col_name = 'nutrient'):
    cv_list = np.std(df, axis=0, ddof=1) / np.mean(df, axis=0)
    cv_df = cv_list.rename_axis(col_name).reset_index(name='variation_coefficient').sort_values('variation_coefficient',ascending = False)
    return cv_df

def get_z_scores(df,col):
    df[f'z_score_{col}'] = (df[col] - df[col].mean())/df[col].std(ddof=1)
    return df

def get_outlier_count(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    output_df = (((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum()).rename_axis('nutrient').reset_index(name='outlier_count').sort_values('outlier_count',ascending = False)
    return output_df

