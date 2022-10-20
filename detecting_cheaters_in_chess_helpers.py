import numpy as np
import pandas as pd
import re

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
        missing_cols = [i for i in cols if i not in df.columns]
        if len(missing_cols)!=0:
            for col in [f'Column {i} not found' for i in missing_cols]:
                print(col)
            return False
        else:
            return True


    def dtype_checker(df, schema):
        wrong_dtypes = [[i, j, df[i].dtype] for i, j in schema.items() if i in df.columns and df[i].dtype!=j]
        if len(wrong_dtypes)!=0:
            for row in [f'Column {k[0]} does not match datatype {k[1]}, is actually {k[2]}' for k in wrong_dtypes]:
                print(row)
            return False
        else:
            return True
        
    print(f'For year: {df.Date[0].year}')    
    schema_check = schema_checker(df, expected_schema.keys())
    dtype_check = dtype_checker(df, expected_schema)
    return print(f'Expected columns present: {schema_check} \nExpected dtypes present: {dtype_check}')


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
    no_nan = all(
        df[[x for x in df.columns if x not in list_of_cols]].notna()
    ) # returns True if all values are not NaN
    if no_nan == True:
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


def keep_decisive_results(df):
    temp_ = df.copy()
    # Computer wins as white 
    comp_w_w = ((temp_['Result']=='1-0')&(temp_['WhiteIsComp']==1))
    # Computer wins as black
    comp_b_w = ((temp_['Result']=='0-1')&(temp_['BlackIsComp']==1))
    temp_ = temp_[
        ((temp_['Result']=='1-0')&(temp_['WhiteIsComp']==1))|
        ((temp_['Result']=='0-1')&(temp_['BlackIsComp']==1))]
    return temp_
    

def drop_uneeded_cols(df=None, list_of_cols=None, what_cols=False):
    '''
    By default, this will drop:
    'Event', 'Site', 'Date', 'Round', 'White', 'Black', 'BlackClock', 'FICSGamesDBGameNo', 'Time', 'WhiteClock',
        'Result'
        
            
    Assert that df is dataframe
    
    '''
    
    col_to_drop = [
    'Event', 'Site', 'Date', 'Round', 'White', 'Black', 'BlackClock', 'FICSGamesDBGameNo', 'Time', 'WhiteClock',
    'Result'
    ]
    
    if what_cols==True:
        return(print(col_to_drop))
    elif list_of_cols is not None and df is not None:
        temp_ = df.drop(columns=list_of_cols)
        print(f'Dropped columns: {list_of_cols}\n')
        return temp_
    else:
        assert isinstance(df, pd.DataFrame), "Passed df must be a df"
        temp_ = df.drop(columns=col_to_drop)
        print(f'Dropped columns: {col_to_drop}\n')
        return temp_
    

def split_timeformat(df):
    temp_ = df.copy()
    temp_[['TimeControl_Base', 'TimeControl_Inc']]=[
    (int(str.split(x, sep='+')[0]), int(str.split(x, sep='+')[1])) for x in temp_.TimeControl.values]
    temp_ = temp_.drop(columns=['TimeControl'])
    return temp_
    
    
def clean_pickle(pickle_df):
    chess_checker_function(pickle_df)
    temp_ = change_comp_columns(pickle_df)
    chess_nan_checker(temp_)
    temp_ = keep_decisive_results(temp_)
    print(f'Keeping decisive results...')
    temp_ = drop_uneeded_cols(temp_)
    temp_ = split_timeformat(temp_)
    return temp_


def clean_pickles(list_of_pickle_df):
    list_of_cleaned_pickle_df = []
    for pickle_df in list_of_pickle_df:
        list_of_cleaned_pickle_df.append(clean_pickle(pickle_df))
        
    return list_of_cleaned_pickle_df


def concatenate_cleaned_pickles(list_of_pickle_df):
    temp_ = pd.concat(clean_pickles(list_of_pickle_df)).reset_index(drop=True)
    return temp_
        
     
def X_y_split_simple(df):
    X_ = df.drop(columns=['WhiteIsComp', 'BlackIsComp', 'NoComp'])
    y_ = df.loc[:,['WhiteIsComp', 'BlackIsComp', 'NoComp']]
    return X_, y_


def y_convert_to_ohe_vec(y):
    '''
    Note: Loss will have to be 'categorical_crossentropy' if y's are one-hot-encoded, 'sparse_categorical_crossentropy' if y's are integers.
    
    If all games contain a cheater, return integers for y where:
        0 indicates white was cheating
        1 indicates black was cheating
        
    If any games contain no cheaters, return one-hot-encoded arrays for y where:
        [1,0,0] indicates white was cheating
        [0,1,0] indicates black was cheating
        [0,0,1] indicates no-one was cheating
        
        
    Have a check for explicit 'options=3' where user can input options value, or check len(y.columns)==3
    '''
    if len(y.columns)==3:
        if all(y.NoComp==0):
            y_ = pd.DataFrame(np.where(y.WhiteIsComp==1, 0, 1), columns=['0_WhiteIsComp_1_BlackIsComp'])
            return y_
        else:
            y_ = pd.DataFrame([[row] for row in y.values], columns=['WhiteIsComp_BlackIsComp_NoComp'])
            return y_        
    else:
        return(print('Function only currently supporting y having WhiteIsComp, BlackIsComp, and NoComp'))