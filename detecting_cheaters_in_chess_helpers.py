import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, plot_confusion_matrix
import matplotlib.pyplot as plt


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

    def check_comp_columns(df):
        if 'BlackIsComp' not in df.columns:
            print('BlackIsComp not present; correting with NaNs')
            df['BlackIsComp'] = np.NaN
        else:
            pass

        if 'WhiteIsComp' not in df.columns:
            print('WhiteIsComp not present; correcting with NaNs')
            df['WhiteIsComp'] = np.NaN
        else:
            pass
        return df
       
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
        
    print(f'For year: {df.Date.iloc[-10].year}')
    df = check_comp_columns(df)
    schema_check = schema_checker(df, expected_schema.keys())
    dtype_check = dtype_checker(df, expected_schema)
    print(f'Expected columns present: {schema_check} \nExpected dtypes present: {dtype_check}')
    return df


def change_comp_columns(df):
    ## make copy df

    nocomp_df = df.copy()
    
    ## check number of unique in blackiscomp and whiteiscomp (should be 2)
    assert all([nocomp_df[x].nunique(dropna=False)<3 for x in ['WhiteIsComp', 'BlackIsComp']]), 'More than two unique values in a XIsComp column (including nan)'
    
#     ## check one of the unique values are 'Yes'
#     assert all(['Yes' in nocomp_df[x].unique() for x in ['WhiteIsComp', 'BlackIsComp']]), 'Missing "Yes" in one of the XIsComp column'

    ## assign 1 and 0 to blackiscomp and whiteiscomp

    nocomp_df['WhiteIsComp'] = np.where(nocomp_df['WhiteIsComp']=='Yes', np.int8(1), np.int8(0))
    nocomp_df['BlackIsComp'] = np.where(nocomp_df['BlackIsComp']=='Yes', np.int8(1), np.int8(0))

    ## make new 'nocomp' column which depends on blackiscomp==1 and whiteiscomp==1

    nocomp_df['NoComp'] = np.where(next(zip(nocomp_df['WhiteIsComp'], nocomp_df['BlackIsComp']))==(0, 0), np.int8(1), np.int8(0))
    
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
    # No cheater, no draw
    no_comp_wb_w = ((temp_['Result']!='1/2-1/2')&(temp_['NoComp']==1))
#     temp_ = temp_[
#         ((temp_['Result']=='1-0')&(temp_['WhiteIsComp']==1))|
#         ((temp_['Result']=='0-1')&(temp_['BlackIsComp']==1))|
#         ((temp_['Result']!='1/2-1/2')&(temp_['NoComp']==1))].reset_index(drop=True)
    temp_ = temp_[
        comp_w_w|
        comp_b_w|
        no_comp_wb_w].reset_index(drop=True)
    return temp_
    

def drop_uneeded_cols(df=None, list_of_cols=None, what_cols=False):
    '''
    By default, this will drop:
    'Event', 'Site', 'Date', 'Round', 'White', 'Black', 'BlackClock', 'FICSGamesDBGameNo', 'Time', 'WhiteClock',
        'Result'
    
    Set list_of_cols='None' to drop no columns.
            
    Assert that df is dataframe
    
    '''
    
    col_to_drop = [
    'Event', 'Site', 'Date', 'Round', 'White', 'Black', 'BlackClock', 'FICSGamesDBGameNo', 'Time', 'WhiteClock',
    'Result'
    ]
    
    if what_cols==True:
        return(print(col_to_drop))
    elif list_of_cols=='None' and df is not None:
        print(f'Dropping no columns...')
        return temp_
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
#     temp_[['TimeControl_Base', 'TimeControl_Inc']]=[
#     (int(str.split(x, sep='+')[0]), int(str.split(x, sep='+')[1])) for x in temp_.TimeControl]
    temp_ = temp_.drop(columns=['TimeControl'])
    return temp_
   
    
def clean_pickle(pickle_df, list_of_cols=None):
    chess_checker_function(pickle_df)
    temp_ = change_comp_columns(pickle_df)
    chess_nan_checker(temp_)
    temp_ = keep_decisive_results(temp_)
    print(f'Keeping decisive results...')
    temp_ = drop_uneeded_cols(temp_, list_of_cols)
    temp_ = split_timeformat(temp_)
    return temp_


def clean_pickles(list_of_pickle_df, list_of_cols=None):
    list_of_cleaned_pickle_df = []
    for pickle_df in list_of_pickle_df:
        list_of_cleaned_pickle_df.append(clean_pickle(pickle_df, list_of_cols))
        
    return list_of_cleaned_pickle_df


def drop_duplicates(temp_):
    pre_shape = temp_.shape
    print(f'Shape pre-drop: {pre_shape}')
    print(f'Dropping duplicates...')
    temp_ = temp_[~temp_.duplicated(subset=[col for col in temp_.columns if col not in ['emt', 'moves']])]
    post_shape = temp_.shape
    print(f'Shape post-drop: {post_shape}\n Duplicates dropped: {pre_shape[0]-post_shape[0]}')        
    return temp_


def does_plycount_match_moves(df):
    '''
    This checks that the number in the PlyCount cell matches the length of the moves list
    
    Should assert that df is df, PlyCount is a column and one of ints, moves is a column  and one of lists
    '''
    if all([len(row[0])==row[1] for row in zip(df.moves, df.PlyCount)]):
        print('All PlyCount values match the length of moves-list')
        return True
    else:
        print('Not all PlyCount values match the length of moves-list')
        return False
    

def concatenate_cleaned_pickles(list_of_pickle_df, list_of_cols=None, drop_dups=True):
    temp_ = pd.concat(clean_pickles(list_of_pickle_df, list_of_cols)).reset_index(drop=True)
    does_plycount_match_moves(temp_)
    if drop_dups==True:
        temp_ = drop_duplicates(temp_)
        return temp_
    else:
        print('Not dropping duplicates...')
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
            print('All games have a cheater: use "sparse_categorical_crossentropy"')
            return y_
        else:
            y_ = pd.DataFrame([[row] for row in y.values], columns=['WhiteIsComp_BlackIsComp_NoComp'])
            print('Some games contain no cheaters: use "categorical_crossentropy"')
            return y_
    else:
        return(print('Function only currently supporting y having WhiteIsComp, BlackIsComp, and NoComp'))
    
    
def y_convert_to_ints(y, expecting_binary=False):
    '''
    Note: NN loss will have to be 'sparse_categorical_crossentropy' if y's are integers.
    
    Return integers for y where:
        0 indicates white was cheating
        1 indicates black was cheating
        2 indicates noone was cheating
        
    The corresponding 'row' value will be:
        [1,0,0] indicates white was cheating
        [0,1,0] indicates black was cheating
        [0,0,1] indicates no-one was cheating
        
        
    Have a check for explicit 'options=3' where user can input options value, or check len(y.columns)==3
    '''
    if len(y.columns)==3:
        if expecting_binary==True and all(y.NoComp==0):
            y_ = pd.DataFrame(np.where(y.WhiteIsComp==1, 0, 1), columns=['0_WhiteIsComp_1_BlackIsComp'])
            print('All games have a cheater')
            return y_
        else:
            y_ = pd.DataFrame([np.int8(0) if all(row==[1,0,0]) else np.int8(1) if all(row==[0,1,0]) else np.int8(2) for row in y.values], columns=['WhiteIsComp_BlackIsComp_NoComp'])
            if all(y.NoComp==0):
                print('All games have a cheater')
            else:
                print('Some games contain no cheaters')
            return y_
    else:
        return(print('Function only currently supporting y having WhiteIsComp, BlackIsComp, and NoComp'))
    
    
def OHE_ECO(X_train, X_test):   
    ohe_ = OneHotEncoder(sparse=False, dtype=np.int8(), handle_unknown='ignore')
    X_train_temp_ = pd.merge(
        left=X_train.drop(columns=['ECO']).reset_index(drop=True),
        right=pd.DataFrame(ohe_.fit_transform(X_train[['ECO']]),
                           columns=ohe_.get_feature_names_out()),
        how='left',
        left_index=True,
        right_index=True)
    
    X_test_temp_ = pd.merge(
        left=X_test.drop(columns=['ECO']).reset_index(drop=True),
        right=pd.DataFrame(ohe_.transform(X_test[['ECO']]),
                           columns=ohe_.get_feature_names_out()),
        how='left',
        left_index=True,
        right_index=True)
    return X_train_temp_, X_test_temp_, ohe_


def class_model_eval_logreg(class_model_, X_train_, X_test_, y_train_, y_test_, digits_=4):
    '''
    Note: Intended for Logistic Regression (may work for others), but not a DT or Random Forest
    
    '''
    # print the accuracy on the training and test set
    print(f'The accuracy score on the training data is: {class_model_.score(X_train_, y_train_)}')
    print(f'The accuracy score on the testing data is: {class_model_.score(X_test_, y_test_)}')
    
    # plot the confusion matrix
    plot_confusion_matrix(class_model_, X_test_, y_test_)
    plt.show()
    
    # classification report
    class_model_pred = class_model_.predict(X_test_)
    report_ = classification_report(y_test_, class_model_pred, digits=digits_)
    print(report_)
    
    # model results
    model_results_ = pd.DataFrame(classification_report(y_test_, class_model_pred,
                                                        digits=digits_, output_dict=True)).loc[
        ['precision', 'recall', 'f1-score'],
        ['0', '1', 'accuracy']]
    
    # model coefficients
    model_coeffs_ = pd.DataFrame(data=abs(class_model_.coef_),
                                columns=class_model_.feature_names_in_).T
    
    return report_, model_results_, model_coeffs_
