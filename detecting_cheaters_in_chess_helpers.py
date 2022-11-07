import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import joblib
import chess
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, plot_confusion_matrix, confusion_matrix, make_scorer, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier



def check_comp_columns(df_):
    
    df=df_.copy()
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
        

def chess_checker_function(df_):
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
    
    df=df_.copy()
    print(f'For year: {df.Date.mean().year}')
    df = check_comp_columns(df)
    schema_check = schema_checker(df, expected_schema.keys())
    dtype_check = dtype_checker(df, expected_schema)
    print(f'Expected columns present: {schema_check} \nExpected dtypes present: {dtype_check}')
    return df


def change_comp_columns(df_):
    '''
    Expects there to be WhiteIsComp and BlackIsComp columns, and that 'Yes' indicates which colour is the computer.
    '''
    
    df=df_.copy()

    ## check number of unique in blackiscomp and whiteiscomp (should be 2)
    assert all([df[x].nunique(dropna=False)<3 for x in ['WhiteIsComp', 'BlackIsComp']]), 'More than two unique values in a XIsComp column (including nan)'
    
#     ## check one of the unique values are 'Yes'
#     assert all(['Yes' in nocomp_df[x].unique() for x in ['WhiteIsComp', 'BlackIsComp']]), 'Missing "Yes" in one of the XIsComp column'

    ## assign 1 and 0 to blackiscomp and whiteiscomp

    df['WhiteIsComp'] = np.where(df['WhiteIsComp']=='Yes', np.int8(1), np.int8(0))
    df['BlackIsComp'] = np.where(df['BlackIsComp']=='Yes', np.int8(1), np.int8(0))

    ## make new 'nocomp' column which depends on blackiscomp==1 and whiteiscomp==1

    df['NoComp'] = np.where(next(zip(df['WhiteIsComp'], df['BlackIsComp']))==(0, 0), np.int8(1), np.int8(0))
    
    return df


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


def drop_no_move_games(df_, min_game_len=None):
    '''
    Will drop games that have no moves
    
    If min_game_len is specified, games with fewer ply than min_game_len will be dropped
    '''
    
    df=df_.copy()
    if any([val_==[] for val_ in df.moves]):
        print(f'Some games have no moves \nDropping those games...')
        df = df[[val_!=[] for val_ in df.moves]].reset_index(drop=True)
    else:
        print('All games have moves')

    if min_game_len is not None:
        assert isinstance(min_game_len, int), "min_game_len must be an int"
        assert min_game_len>0, "min_game_len must be a postive int"
        pre_shape=df.shape[0]
        df=df[[len(val_)>=min_game_len for val_ in df.moves]].reset_index(drop=True)
        post_shape=df.shape[0]
        if pre_shape!=post_shape:
            print(f'Dropped {pre_shape-post_shape} games')
        else:
            print(f'No games shorter than {min_game_len} ply')
        return df
    else:
        return df


def any_missing_emt(df):
    if any([val_==[] for val_ in df.emt]):
        return(print('Some games missing emt values...'))
    else:
        return(print('No games missing emt values'))


def keep_decisive_results(df_):
#     temp_ = df.copy()
    df=df_.copy()
    # Computer wins as white 
    comp_w_w = ((df['Result']=='1-0')&(df['WhiteIsComp']==1))
    # Computer wins as black
    comp_b_w = ((df['Result']=='0-1')&(df['BlackIsComp']==1))
    # No cheater, no draw
    no_comp_wb_w = ((df['Result']!='1/2-1/2')&(df['NoComp']==1))
#     temp_ = temp_[
#         ((temp_['Result']=='1-0')&(temp_['WhiteIsComp']==1))|
#         ((temp_['Result']=='0-1')&(temp_['BlackIsComp']==1))|
#         ((temp_['Result']!='1/2-1/2')&(temp_['NoComp']==1))].reset_index(drop=True)
    df = df[
        comp_w_w|
        comp_b_w|
        no_comp_wb_w].reset_index(drop=True)
    return df
    

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
        return(print(f' Default columns to drop are: {col_to_drop}'))
    elif list_of_cols=='None' and df is not None:
        print(f'Dropping no columns...')
        return df
    elif list_of_cols is not None and df is not None:
        assert isinstance(df, pd.DataFrame), "Passed df must be a df"
        assert isinstance(list_of_cols, list), "Passed list_of_cols must be a list of str"
        assert all([type(x)==str for x in list_of_cols]), "Passed list_of_cols must be a list of str"
        temp_ = df.drop(columns=list_of_cols)
        print(f'Dropped columns: {list_of_cols}\n')
        return temp_
    else:
        assert isinstance(df, pd.DataFrame), "Passed df must be a df"
        temp_ = df.drop(columns=col_to_drop)
        print(f'Dropped columns: {col_to_drop}\n')
        return temp_
    

def split_timeformat(df_):
    '''
    Expects there to be a TimeControl column
    '''
    
    df=df_.copy()
    df[['TimeControl_Base', 'TimeControl_Inc']]=[
    (int(str.split(x, sep='+')[0]), int(str.split(x, sep='+')[1])) for x in df.TimeControl.values]
#     temp_[['TimeControl_Base', 'TimeControl_Inc']]=[
#     (int(str.split(x, sep='+')[0]), int(str.split(x, sep='+')[1])) for x in temp_.TimeControl]
    df = df.drop(columns=['TimeControl'])
    return df
   

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

    
def clean_pickle(pickle_df_, list_of_cols=None, filtering_steps=(True, True, True), min_game_len=None):
    '''
    This will clean a pickled df
    
    Deafult value for filtering_steps will do all of them
    
    Pass a tuple for filtering_steps, their order being for dropping no move games, keeping decisive results and dropping uneeded columns
    
    ....
    
    '''
    drop_no_move_game_, keep_decisive_results_, drop_uneeded_cols_ = filtering_steps
    
    pickle_df=pickle_df_.copy()
    pickle_df=chess_checker_function(pickle_df)
    temp_ = change_comp_columns(pickle_df)
    chess_nan_checker(temp_)
    if drop_no_move_game_:
        temp_ = drop_no_move_games(temp_, min_game_len)
    does_plycount_match_moves(temp_)
    any_missing_emt(temp_)
    if keep_decisive_results_:
        temp_ = keep_decisive_results(temp_)
        print(f'Keeping decisive results...')
    if drop_uneeded_cols_:
        temp_ = drop_uneeded_cols(temp_, list_of_cols)
    temp_ = split_timeformat(temp_)
    return temp_


def clean_pickles(list_of_pickle_df, list_of_cols=None, filtering_steps=(True, True, True)):
    list_of_cleaned_pickle_df = []
    for pickle_df in list_of_pickle_df:
        list_of_cleaned_pickle_df.append(clean_pickle(pickle_df, list_of_cols, filtering_steps))
        
    return list_of_cleaned_pickle_df


def drop_duplicates(df_):
    df=df_.copy()
    pre_shape = df.shape
    print(f'Shape pre-drop: {pre_shape}')
    print(f'Dropping duplicates...')
    df = df[~df.duplicated(subset=[col for col in df.columns if col not in ['emt', 'moves']])]
    post_shape = df.shape
    print(f'Shape post-drop: {post_shape}\n Duplicates dropped: {pre_shape[0]-post_shape[0]}')
    df = df.reset_index(drop=True)
    return df


def concatenate_cleaned_pickles(list_of_pickle_df, list_of_cols=None,
                                filtering_steps=(True, True, True) , drop_dups=True):
    '''
    Pass a tuple for filtering_steps, their order being for: dropping no move games, keeping decisive results and dropping uneeded columns
    '''
    temp_ = pd.concat(clean_pickles(list_of_pickle_df, list_of_cols, filtering_steps)).reset_index(drop=True)
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
    

def flat_y(y_train_in, y_test_in):
    '''
    y_train and y_test should be dataframes
    
    Outputs y_train vector and y_test vector (i.e. ({any int},))
    '''
    
    y_train = np.reshape(y_train_in.values, (-1,))
    y_test = np.reshape(y_test_in.values, (-1,))
    
    return y_train, y_test

    
def OHE_ECO(X_train, X_test): 
    '''
    Output: X_train_temp_, X_test_temp_, ohe_
    '''
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


def stan_scale(X_train_, X_test_, list_of_cols=None):
    '''
    Will use standard scaler to fit_transform on the train and transform the test
    
    Outputs X_train_scaled_, X_test_scaled and stan_scal_
    '''
    assert isinstance(list_of_cols, list), 'list_of_cols must be a list'
    assert list_of_cols is not None, 'list_of_cols cannot be None'
    
    X_train_scaled = X_train_.copy() # copies must be made otherwise even the input will be overwritten
    X_test_scaled = X_test_.copy()
    
    stan_scal_ = StandardScaler()
    X_train_scaled[list_of_cols] = stan_scal_.fit_transform(X_train_scaled[list_of_cols])
    
    X_test_scaled[list_of_cols] = stan_scal_.transform(X_test_scaled[list_of_cols])
    
    return X_train_scaled, X_test_scaled, stan_scal_


def drop_emt_moves(df_):
    df=df_.copy()
    df = df.drop(columns=['emt', 'moves'])
    return df


def print_accuracy(class_model_, X_train_, X_test_, y_train_, y_test_):
    # print the accuracy on the training and test set
    return print(f'The accuracy score on the training data is: {class_model_.score(X_train_, y_train_)}\nThe accuracy score on the testing data is: {class_model_.score(X_test_, y_test_)}')   


def class_model_eval_logreg(class_model_, X_train_, X_test_, y_train_, y_test_, num_class=2, digits_=4, has_coeffs=True, is_knn_=False):
    '''
    Note: Intended for Logistic Regression (may work for others), but not a DT or Random Forest
    
    '''
    # print the accuracy on the training and test set
    print_accuracy(class_model_, X_train_, X_test_, y_train_, y_test_)
    
    # plot the confusion matrix
    conf_matr = plot_confusion_matrix(class_model_, X_test_, y_test_)
    plt.show()
    
    # classification report
    class_model_pred = class_model_.predict(X_test_)
    report_ = classification_report(y_test_, class_model_pred, digits=digits_)
    print(report_)
    
    # model results
    num_classes = [f'{i}' for i in range(num_class)]
    num_classes.append('accuracy')
    
    model_results_ = pd.DataFrame(classification_report(y_test_, class_model_pred,
                                                        digits=digits_, output_dict=True)).loc[
        ['precision', 'recall', 'f1-score'],
        num_classes]  
        
    
    # model coefficients
    if has_coeffs:
        model_coeffs_ = pd.DataFrame(data=class_model_.coef_,
                                    columns=class_model_.feature_names_in_).T
        return report_, model_results_, model_coeffs_
    elif is_knn_:
        return report_, model_results_, conf_matr
    else:
        return report_, model_results_


def extract_emt_elements(game_emts):
    times_ = [float(re.search('[0-9]+\.[0-9]+', time_[0]).group()) for time_ in zip(game_emts)]
    return times_


def extract_all_emt_elements(df_):
    '''
    This will receive a dataframe with a column of emt
    This will output the dataframe a new columns: emt_time
    '''
    df=df_.copy()
    df['emt_time']=[extract_emt_elements(game_[0]) for game_ in zip(df.emt)]
    return df


def get_white_emts(list_of_emts):
    '''
    Will receive a list of a games emt values
    '''
    
    white_emts = list_of_emts[::2]
    return white_emts


def get_black_emts(list_of_emts):
    '''
    Will receive a list of a games emt values
    '''
    
    black_emts = list_of_emts[1::2]
    return black_emts


def separate_all_white_and_black_emts(df_, keep_all_calc_cols=False):
    '''
    This will receive a dataframe with a column of emt
    If keep_all_calc_cols is True, the returned dataframe will have all the calculated columns
    '''
    df=df_.copy()
        
    # first extract all the emt elements
    df=extract_all_emt_elements(df)
    
    df[['white_emt', 'black_emt']]=[[get_white_emts(list_of_emts[0]), get_black_emts(list_of_emts[0])] for list_of_emts in zip(df.emt_time)]
    
    # return is dependent on what columns are wanted
    if keep_all_calc_cols:
        return df
    else:
        df=df.drop(columns=['emt_time'])
        return df


def separate_all_white_and_black_average_emts(df_, rounded_dp=5, keep_all_calc_cols=False):
    '''
    This will receive a dataframe with a column of emt
    If keep_all_calc_cols is True, the returned dataframe will have all the calculated columns
    '''
    df=df_.copy()
    
    # first extract all the emt elements
    df=extract_all_emt_elements(df)
    
    df[['av_white_emt', 'av_black_emt']]=[[np.round(np.mean(get_white_emts(list_of_emts[0])), rounded_dp), np.round(np.mean(get_black_emts(list_of_emts[0])), rounded_dp)] for list_of_emts in zip(df.emt_time)]
    
    # return is dependent on what columns are wanted
    if keep_all_calc_cols:
        return df
    else:
        df=df.drop(columns=['emt_time'])
        return df


def how_games_ended(series_of_games_emts):
    '''
    test_ = df_2021_2017_titled_distinv.copy()
    test_['endings'] = how_games_ended(test_.emt)
    test_.head(2)
    '''
    
#     series_of_games_emts = [game_ if game_!=[] else ['none'] for game_ in series_of_games_emts]
#     endings_ = [re.search('[a-zA-Z]{5}.[a-zA-Z]+.[a-zA-Z]*.[a-zA-Z]*', game_[-1]).group() for game_ in series_of_games_emts]
#     endings_ = [re.search('(Black|White).[a-zA-Z]+.[a-zA-Z]*.[a-zA-Z]*', game_[-1]).group() for game_ in series_of_games_emts]
    endings_ = [re.search('(Black|White).[a-zA-Z]+.[a-zA-Z]*.[a-zA-Z]*', game_[-1]).group() if game_!=[] else [] for game_ in series_of_games_emts]
    
    return endings_


def how_games_ended_confirmed_moves_and_emt(series_of_games_emts):
#     series_of_games_emts = [game_ if game_!=[] else ['none'] for game_ in series_of_games_emts]
#     endings_ = [re.search('[a-zA-Z]{5}.[a-zA-Z]+.[a-zA-Z]*.[a-zA-Z]*', game_[-1]).group() for game_ in series_of_games_emts]
#     endings_ = [re.search('(Black|White).[a-zA-Z]+.[a-zA-Z]*.[a-zA-Z]*', game_[-1]).group() for game_ in series_of_games_emts]
    endings_ = [re.search('(Black|White).[a-zA-Z]+.[a-zA-Z]*.[a-zA-Z]*', game_[-1]).group() for game_ in series_of_games_emts]
    
    return endings_


def evaluate_game(game):
    # Loop for moves in a single game
    evaluations_ = []
    board=chess.Board()
    for move in game:
        evalution_ = engine.analyse(board, limit, multipv='5')
        board.push_san(move)

        evaluations_.append(evalution_)
    
    
    return evaluations_


def evaluate_games(df, save_rate=1000, path='./'): # risky to use function on many games in case something goes wrong
    list_of_evaluations = []
    game_count = 0
    
    for game in zip(df.moves):    
        try:
            game_eval_ = evaluate_game(game[0])
            list_of_evaluations.append(game_eval_)

            game_count+=1

            if game_count%save_rate==0:
                print(f'{game_count} games completed\nSaving now...')
                joblib.dump(list_of_evaluations, 
                            f'{path}{game_count}_.pkl',
                           compress=3)
                t=time.localtime()[0:6]
                print(f'Saved at {t[0]}/{t[1]}/{t[2]} {t[3]}:{t[4]}:{t[5]}')
            elif game_count%100==0:
                print(f'{game_count}')
                t=time.localtime()[0:6]
                print(f'At {t[0]}/{t[1]}/{t[2]} {t[3]}:{t[4]}:{t[5]}')
            else:
                pass
            
        except KeyboardInterrupt:
            print('Keyboard Interrupt')
            print(f'{game_count}')
            return list_of_evaluations
#             break
    
        except:
            list_of_evaluations.append(['Error occured'])

            game_count+=1        

            print(f'Error occured on game {game_count}')
        
    return list_of_evaluations


def get_abs_elo_diff(df_):
    '''
    Pass in a df with BlackElo and WhiteElo and get back the absolute value of elo difference
    '''
    df=df_.copy()
    df['abs_elo_diff']=[abs(row[0]-row[1]) for row in df[['WhiteElo', 'BlackElo']].values]
    
    return df


def get_rel_elo_diff(df_):
    '''
    Pass in a df with BlackElo and WhiteElo and get back the relative value of elo difference
        A positive number means that WhiteElo>BlackElo
    '''
    
    df=df_.copy()
    df['abs_elo_diff']=[row[0]-row[1] for row in df[['WhiteElo', 'BlackElo']].values]
    
    return df


def keep_rated_games(df_):
    df = df_[[' rated' in x for x in df_.Event]]
    
    return df


def keep_time_length_games_greater_than(df_, time_control_base_minimum=3600):
    df = df_[df_['TimeControl_Base']>=time_control_base_minimum]
    
    return df


def train_default_logreg_model(X_train_scaled_, y_train_, max_iter_=10000, n_jobs_=-1,
                               path=None, name='', suffix='_logreg.pkl', compress_=9):
    '''
    Will train a logistic regression model using default settings except for max_iter_ and n_jobs_
    y_train_ must be a vector (i.e. ({any int},) array)
    path is the path to automatically save the model if desired
    
    Outputs the model (logreg)
    '''
    
    logreg = LogisticRegression(max_iter=max_iter_, n_jobs=n_jobs_)
    logreg.fit(X_train_scaled_, y_train_)
    if path is not None:
        assert isinstance(name, str), 'path must be a string if given as argument'
        joblib.dump(logreg, f'{path}{name}{suffix}', compress=compress_)
        print(f'Saved {name}{suffix} at {path}')
        
    return logreg


def binary_logreg_ranked_coefs(logreg_model, top_n=15):
    '''
    top_n is how many top coefs to show
    
    Outputs a dataframe of the top coefficients in a binary logistic regression classification
    '''
    log_reg_coefficients = pd.DataFrame(
        data=abs(logreg_model.coef_), columns=logreg_model.feature_names_in_)

    top_coeffs = log_reg_coefficients.T.rename(columns={0: 'coefficient_weights'}).sort_values(
        by='coefficient_weights', ascending=False)
    display(top_coeffs.head(top_n))
    
    return top_coeffs


def train_default_knn_model(X_train_scaled_, y_train_, n_jobs_=-1,
                               path=None, name='', suffix='_knn.pkl', compress_=9):
    '''
    Will train a knn model using default settings except for n_jobs_
    y_train_ must be a vector (i.e. ({any int},) array)
    path is the path to automatically save the model if desired
    
    Outputs the model (knn)
    '''
    
    knn = KNeighborsClassifier(n_jobs=n_jobs_)
    knn.fit(X_train_scaled_, y_train_)
    if path is not None:
        assert isinstance(name, str), 'path must be a string if given as argument'
        joblib.dump(knn, f'{path}{name}{suffix}', compress=compress_)
        print(f'Saved {name}{suffix} at {path}')
        
    return knn
    
    
def train_default_dt_model(X_train_, y_train_, max_features_='sqrt', 
                               path=None, name='', suffix='_dt.pkl', compress_=9):
    '''
    Will train a dt model using default settings except for max_features_
    y_train_ must be a vector (i.e. ({any int},) array)
    path is the path to automatically save the model if desired
    
    Outputs the model (dt)
    '''
    
    dt = DecisionTreeClassifier(max_features=max_features_)
    dt.fit(X_train_, y_train_)
    if path is not None:
        assert isinstance(name, str), 'path must be a string if given as argument'
        joblib.dump(dt, f'{path}{name}{suffix}', compress=compress_)
        print(f'Saved {name}{suffix} at {path}')
    return dt   
    
    
def train_default_rf_model(X_train_, y_train_, n_jobs_=-1,
                               path=None, name='', suffix='_rf.pkl', compress_=9):
    '''
    Will train a rf model using default settings except for n_jobs_
    y_train_ must be a vector (i.e. ({any int},) array)
    path is the path to automatically save the model if desired
    
    Outputs the model (rf)
    '''
    
    rf = RandomForestClassifier(n_jobs=n_jobs_)
    rf.fit(X_train_, y_train_)
    if path is not None:
        assert isinstance(name, str), 'path must be a string if given as argument'
        joblib.dump(rf, f'{path}{name}{suffix}', compress=compress_)
        print(f'Saved {name}{suffix} at {path}')
    return rf 
    
    
def train_default_xgboost_model(X_train_, y_train_, n_jobs_=-1,
                               path=None, name='', suffix='_xgb.pkl', compress_=9):
    '''
    Will train a xgb model using default settings except j_jobs_
    y_train_ must be a vector (i.e. ({any int},) array)
    path is the path to automatically save the model if desired
    
    Outputs the model (xgb)
    '''
    
    xgb = XGBClassifier(n_jobs=n_jobs_)
    xgb.fit(X_train_, y_train_)
    if path is not None:
        assert isinstance(name, str), 'path must be a string if given as argument'
        joblib.dump(xgb, f'{path}{name}{suffix}', compress=compress_)
        print(f'Saved {name}{suffix} at {path}')
    return xgb  
    

def tree_feature_importance(class_model_, X_train_, top_n=15):
    '''
    class_model_ should be a model utilising trees
    X_train_ should be (un)scaled as the tree model 
    top_n should be an int
    '''
    feature_df_tree = pd.DataFrame({"feature_importance": class_model_.feature_importances_}, index=X_train_.columns)
    feature_df_tree = feature_df_tree.sort_values(by='feature_importance', ascending=False).head(top_n)

    return feature_df_tree


def rf_trees_and_forest_strength(rf_model_, X_train_, X_test_, y_train_, y_test_):
    '''
    Run with ; at the end if you're not saving to the output variables of av_dt_train, av_dt_test, rf_train, rf_test
    '''
    
    decision_tree_train_scores = []
    decision_tree_test_scores = []
    
    for sub_tree in rf_model_.estimators_:
        decision_tree_train_scores.append(sub_tree.score(X_train_, y_train_))
        decision_tree_test_scores.append(sub_tree.score(X_test_, y_test_))
        
    av_dt_train = np.mean(decision_tree_train_scores)
    av_dt_test = np.mean(decision_tree_test_scores)
    
    rf_train = rf_model_.score(X_train_, y_train_)
    rf_test = rf_model_.score(X_test_, y_test_)
    
    
    print("Performance on fitted data:")
    print(f"Average Decision Tree: {av_dt_train}")
    print(f"Random Forest: {rf_train}\n")
    
        

    print("Performance on Test data:")
    print(f"Average Decision Tree: {rf_train}")
    print(f"Random Forest: {rf_test}")
    
    return (av_dt_train, av_dt_test, rf_train, rf_test)   


def get_best_moves_in_pos(pos_):
    '''
    Extracts the best moves and their respective scores
    
    pos_ should be a list
    
    Outputs a list
    '''
    
    top_in_pos_ = [[move_['score'], chess.Move.uci(move_['pv'][0])] for move_ in pos_]
    
    return top_in_pos_


def get_best_moves_in_game(game_):
    '''
    Extracts the best moves and their respective scores for every position in the game
    
    game_ should be a list
    
    Outputs a list
    '''
    
    top_for_pos_in_game_ = [get_best_moves_in_pos(pos_) for pos_ in game_]
    
    return top_for_pos_in_game_
    

def get_best_moves_in_all_games(list_of_games_, is_list_=True):
    '''
    Extracts the best moves and their respective scores for every position in each game in list_of_games_
    
    list_of_games_ should be a list or series and the input arguments set accordingly
    
    Outputs a list
    '''
    
    if is_list_:
        top_for_games_ = [get_best_moves_in_game(game_) if game_!=['Error occured'] else ['Error occured'] for game_ in list_of_games_]
    else:
        top_for_games_ = [get_best_moves_in_game(game_) if game_!='Error occured' else ['Error occured'] for game_ in list_of_games_]
    
    return top_for_games_


def impose_elo_bounding(df_, lower_bound_=1600, upper_bound_=2857, reset_index_=False):
    '''
    Given a df_, this will return a dataframe where the games are bounded between the lower and upper bound (i.e. greater than or equal to lower bound and less than or equal to upper bound)
    
    If reset_index_ is True, the returned dataframe will have had its index reset
    '''
    
    df_elo_bounded_ = df_[
        (df_.WhiteElo>=lower_bound_) & (df_.WhiteElo<=upper_bound_) &\
        (df_.BlackElo>=lower_bound_) & (df_.BlackElo<=upper_bound_)]
    
    if reset_index_:
        df_elo_bounded_ = df_elo_bounded_.reset_index(drop=True)
        return df_elo_bounded_
    else:
        return df_elo_bounded_
    

def impute_RD_values(df_):
    '''
    Imputes RD values of 'na' to 0
    '''
    df=df_.copy()
    df[['BlackRD', 'WhiteRD']] = np.array([
    [np.int8(0) if y=='na' else np.float16(y) for y in x] for x in df[['BlackRD', 'WhiteRD']].values])
    
    return df


def convert_game_moves_to_uci(game_moves_):
    '''
    Converts one game's moves to uci format
    game_moves_ should be a list (i.e. game.moves)
    '''
    
    board=chess.Board()
    uci_moves_ = [chess.Move.uci(board.push_san(move_[0])) for move_ in zip(game_moves_)]
    
    return uci_moves_


def convert_all_game_moves_to_uci(df_):
    '''
    This will return a dataframe with a new column of uci_moves
    dataframe df_ should have a column of "moves"
    '''
    
    df=df_.copy()
    df['uci_moves'] = [convert_game_moves_to_uci(game_[0]) for game_ in zip(df.moves)]
    
    return df


def get_eval_top_move_for_game(game_eval_):
    '''
    Extracts the top move for every position in the game's evaluation list
    '''
    
    top_move_list_ = [pos_[0][0][1] for pos_ in zip(game_eval_)]
    
    return top_move_list_


def get_eval_top_move_for_all_games(df_):
    '''
    This will return a dataframe with a new column of top_moves
    dataframe df_ should have a column of "eval_"
    '''
    
    df=df_.copy()
    df['top_uci_moves'] = [get_eval_top_move_for_game(game_[0]) for game_ in zip(df.eval_)]
    
    return df


def get_white_moves(list_of_moves):
    '''
    Will receive a list of a game's moves values
    '''
    
    white_moves = list_of_moves[::2]
    return white_moves   


def get_black_moves(list_of_moves):
    '''
    Will receive a list of a game's moves values
    '''
    
    black_moves = list_of_moves[1::2]
    return black_moves  


def separate_all_white_and_black_moves(df_, is_uci=False, include_top_moves=False):
    '''
    This will receive a dataframe with a column of moves or uci_moves
    This will output the dataframe with two new columns: white_(uci_)moves and black_(uci_)moves
    '''
    df=df_.copy()
    if is_uci:
        df[['white_uci_moves', 'black_uci_moves']]=[[get_white_moves(list_of_moves[0]), get_black_moves(list_of_moves[0])] for list_of_moves in zip(df.uci_moves)]
    else:
        df[['white_moves', 'black_moves']]=[[get_white_moves(list_of_moves[0]), get_black_moves(list_of_moves[0])] for list_of_moves in zip(df.moves)]
    
    if include_top_moves:
        df[['white_top_uci_moves', 'black_top_uci_moves']]=[[get_white_moves(list_of_moves[0]), get_black_moves(list_of_moves[0])] for list_of_moves in zip(df.top_uci_moves)]
    else:
        pass

    return df


def percent_of_top_moves_played_in_game_by_white(game_):
    '''
    Expects game_ to have 'white_uci_moves' and 'white_top_uci_moves'
    '''
    
    # Percent Of Top Moves Played By White
    move_pair_comparison_ = [move_pair_[0]==move_pair_[1] for move_pair_ in zip(game_.white_uci_moves, game_.white_top_uci_moves)]
    potmpbw_ = pd.Series(move_pair_comparison_).value_counts(normalize=True)
    if len(potmpbw_)==2:
        potmpbw_ = np.round(pd.Series(move_pair_comparison_).value_counts(normalize=True).loc[True], 6)
    else:
        try:
            potmpbw_ = np.round(pd.Series(move_pair_comparison_).value_counts(normalize=True).loc[True], 6) 
        except:
            potmpbw_ = 0

    return potmpbw_


def percent_of_top_moves_played_in_game_by_black(game_):
    '''
    Expects game_ to have 'black_uci_moves' and 'black_top_uci_moves'
    '''
    
    # Percent Of Top Moves Played By Black
    move_pair_comparison_ = [move_pair_[0]==move_pair_[1] for move_pair_ in zip(game_.black_uci_moves, game_.black_top_uci_moves)]
    potmpbb_ = pd.Series(move_pair_comparison_).value_counts(normalize=True)
    if len(potmpbb_)==2:
        potmpbb_ = np.round(pd.Series(move_pair_comparison_).value_counts(normalize=True).loc[True], 6)
    else:
        try:
            potmpbb_ = np.round(pd.Series(move_pair_comparison_).value_counts(normalize=True).loc[True], 6) 
        except:
            potmpbb_ = 0

    return potmpbb_
    

def percent_of_top_moves_played_in_game_by_white_and_black(game_):
    
    potmpbw_ = percent_of_top_moves_played_in_game_by_white(game_)
    potmpbb_ = percent_of_top_moves_played_in_game_by_black(game_)
    
    return potmpbw_, potmpbb_


def percent_of_top_moves_played_in_all_games_by_white_and_black(df_, keep_all_calc_cols=False):
    '''
    Expects df_ to have columns for uci_moves and top_uci_moves
    
    If keep_all_calc_cols is True, the returned dataframe will have all the calculated columns
    '''
    
    df=df_.copy()
    
    # first separate the moves of white and black
    df=separate_all_white_and_black_moves(df, is_uci=True, include_top_moves=True)
    
    # find the percentage of top moves played by white and black
    df[['white_played_perc_top_move', 'black_played_perc_top_move']] = [
    percent_of_top_moves_played_in_game_by_white_and_black(
        pd.Series(game_[0], index=df.columns)) for game_ in zip(df.values)]
    
    # return is dependent on what columns are wanted
    if keep_all_calc_cols:
        return df
    else:
        df=df.drop(columns=['white_uci_moves', 'white_top_uci_moves', 'black_uci_moves', 'black_top_uci_moves'])
        return df
    
