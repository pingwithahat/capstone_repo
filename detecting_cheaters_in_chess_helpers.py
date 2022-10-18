import numpy as np
import pandas as pd

def chess_checker_function(df):
    expected_schema = {
        'Event': 'object',
        'Site': 'object',
        'Date': 'datetime64[ns]',
        'Round': 'object',
        'White': 'object',
        'Black': 'object',
        'Result': 'object',
        'BlackClock': 'object',
        'BlackElo':  'int64',
        'BlackIsComp': 'object',
        'BlackRD': 'float64',
        'ECO': 'object',
        'FICSGamesDBGameNo':  'int64',
        'PlyCount':  'int64',
        'Time': 'object',
        'TimeControl': 'object',
        'WhiteClock': 'object',
        'WhiteElo':  'int64',
        'WhiteRD': 'float64',
        'emt': 'object',
        'moves': 'object',
        'WhiteIsComp': 'object'  
    }

    def schema_checker(df, cols):
#         for i in cols:
#             if i not in df.columns:
#                 print(f'Column {i} not found')
# #                 return False
#         return True
        missing_cols = [i for i in cols if i not in df.columns]
        if len(missing_cols)!=0:
            print(f'Column {i} not found' for i in missing_cols)
            return False
        else:
            return True


    def dtype_checker(df, schema):
        wrong_dtypes = [[i, j] for i, j in schema.items() if i in df.columns and df[i].dtype!=j]
        if len(wrong_dtypes)!=0:
            for row in [f'Column {k[0]} does not match datatype {k[1]}' for k in wrong_dtypes]:
                print(row)
            return False
        else:
            return True
        
        
    schema_check = schema_checker(df, expected_schema.keys())
    dtype_check = dtype_checker(df, expected_schema)
    return print(f'Expected columns: {schema_check} \nExpected dtypes: {dtype_check}')


def change_comp_columns(df):
    ## make copy df

    nocomp_df = df.copy()
    
    ## check number of unique in blackiscomp and whiteiscomp (should be 2)
    assert all([nocomp_df[x].nunique(dropna=False)==2 for x in ['WhiteIsComp', 'BlackIsComp']]), 'More than two unique values in a XIsComp column (including nan)'
    
    ## check one of the unique values are 'Yes'
    assert all(['Yes' in nocomp_df[x].unique() for x in ['WhiteIsComp', 'BlackIsComp']]), 'Missing "Yes" in one of the XIsComp column'

    ## assign 1 and 0 to blackiscomp and whiteiscomp

    nocomp_df['WhiteIsComp'] = np.where(nocomp_df['WhiteIsComp']=='Yes', np.int8(1), np.int8(0))
    nocomp_df['BlackIsComp'] = np.where(nocomp_df['BlackIsComp']=='Yes', np.int8(1), np.int8(0))

    ## make new 'nocomp' column which depends on blackiscomp==1 and whiteiscomp==1

    nocomp_df['NoComp'] = np.where(zip(nocomp_df['WhiteIsComp'], nocomp_df['BlackIsComp'])==[0, 0], np.int8(1), np.int8(0))
    
    return nocomp_df


def chess_nan_checker(df, list_of_cols=['BlackIsComp', 'WhiteIsComp']): 
    '''
    checks for nan values in the dataframe

    Args:
        df (DataFrame): Pandas dataframe
        list_of_cols (List): List of strings indicating what columns to ignore

    Returns:
        summary (String): returns a summary of nan values
    '''

    assert all([col in df.columns for col in list_of_cols]), f'Passed dataframe does not contain {list_of_cols}'   
    
    # are there any nan values?
    any_nan = all(
        df[[x for x in df.columns if x not in list_of_cols]].notna()
    ) # returns True if all values are not NaN
    if any_nan == True:
        return print(f"This dataframe has 0 NaN values in columns: {list(df[[x for x in df.columns if x not in list_of_cols]].columns)}")
        
    # how many?
    nan_series = df.isna().sum()
    nan_num = nan_series.sum()

    # which variables do they come from
    nan_cols = list(nan_series[nan_series > 0].index)

    # any variables with more than 50% of data missing?
    big_nan_cols = list(nan_series[nan_series/len(df) > 0.5])
    
    #summary can be all the info about nans
    summary = f'This dataframe has {nan_num} NaN values'

    if len(nan_cols) > 0:
        summary += f'\nThe NaN values come from: {nan_cols}'

    if len(big_nan_cols) > 0:
        summary += f'\nMore than 50% of the data is missing from: {big_nan_cols}'

    return print(summary)