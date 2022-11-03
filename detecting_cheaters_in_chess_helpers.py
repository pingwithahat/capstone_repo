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
        
#     print(f'For year: {df.Date.iloc[-10].year}')
    print(f'For year: {df.Date.mean().year}')
    df = check_comp_columns(df)
    schema_check = schema_checker(df, expected_schema.keys())
    dtype_check = dtype_checker(df, expected_schema)
    print(f'Expected columns present: {schema_check} \nExpected dtypes present: {dtype_check}')
    return df


def change_comp_columns(df):
    '''
    Expects there to be WhiteIsComp and BlackIsComp columns, and that 'Yes' indicates which colour is the computer.
    '''
   
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


def drop_no_move_games(df):
    if any([val_==[] for val_ in df.moves]):
        print(f'Some games have no moves \nDropping those games...')
        temp_ = df[[val_!=[] for val_ in df.moves]].reset_index(drop=True)
        return temp_
    else:
        print('All games have moves')
        return df


def any_missing_emt(df):
    if any([val_==[] for val_ in df.emt]):
        return(print('Some games missing emt values...'))
    else:
        return(print('No games missing emt values'))


def keep_decisive_results(df):
#     temp_ = df.copy()
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
    '''
    Expects there to be a TimeControl column
    '''
    
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

    
def clean_pickle(pickle_df, list_of_cols=None, filtering_steps=(True, True, True)):
    '''
    This will clean a pickled df
    
    Deafult value for filtering_steps will do all of them
    
    Pass a tuple for filtering_steps, their order being for dropping no move games, keeping decisive results and dropping uneeded columns
    
    ....
    
    '''
    drop_no_move_game_, keep_decisive_results_, drop_uneeded_cols_ = filtering_steps
    
    chess_checker_function(pickle_df)
    temp_ = change_comp_columns(pickle_df)
    chess_nan_checker(temp_)
    if drop_no_move_game_:
        temp_ = drop_no_move_games(temp_)
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


def drop_duplicates(df):
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


def print_accuracy(class_model_, X_train_, X_test_, y_train_, y_test_):
    # print the accuracy on the training and test set
    return print(f'The accuracy score on the training data is: {class_model_.score(X_train_, y_train_)}\nThe accuracy score on the testing data is: {class_model_.score(X_test_, y_test_)}')
    


def class_model_eval_logreg(class_model_, X_train_, X_test_, y_train_, y_test_, digits_=4, has_coeffs=True):
    '''
    Note: Intended for Logistic Regression (may work for others), but not a DT or Random Forest
    
    '''
    # print the accuracy on the training and test set
    print_accuracy(class_model_, X_train_, X_test_, y_train_, y_test_)
    
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
    if has_coeffs:
        model_coeffs_ = pd.DataFrame(data=abs(class_model_.coef_),
                                    columns=class_model_.feature_names_in_).T
        return report_, model_results_, model_coeffs_
    else:
        return report_, model_results_


def extract_emt_elements(game_emts):
    times_ = [float(re.search('[0-9]+\.[0-9]+', time_).group()) for time_ in game_emts]
    return times_


def extract_all_emt_elements(series_of_games_emts):
    times_ = [extract_emt_elements(game_) for game_ in series_of_games_emts]
    return times_


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


def separate_all_white_and_black_emts(df):
    '''
    This will receive a dataframe with a column of emt_time (Note: this is not the same as emt, emt_time has only float values)
    This will output the dataframe with two new columns: white_emt and black_emt
    '''
    
    df[['white_emt', 'black_emt']]=[[get_white_emts(list_of_emts[0]), get_black_emts(list_of_emts[0])] for list_of_emts in zip(df.emt_time)]
    
    return df


def separate_all_white_and_black_average_emts(df, rounded_dp=5):
    '''
    This will receive a dataframe with a column of emt_time (Note: this is not the same as emt, emt_time has only float values)
    This will output the dataframe with two new columns: av_white_emt and av_black_emt
    '''
    
    df[['av_white_emt', 'av_black_emt']]=[[np.round(np.mean(get_white_emts(list_of_emts[0])), rounded_dp), np.round(np.mean(get_black_emts(list_of_emts[0])), rounded_dp)] for list_of_emts in zip(df.emt_time)]
    
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


def get_abs_elo_diff(df):
    '''
    Pass in a df with BlackElo and WhiteElo and get back the absolute value of elo difference
    '''
    
    df['abs_elo_diff']=[abs(row[0]-row[1]) for row in df[['WhiteElo', 'BlackElo']].values]
    
    return df


def get_rel_elo_diff(df):
    '''
    Pass in a df with BlackElo and WhiteElo and get back the relative value of elo difference
        A positive number means that WhiteElo>BlackElo
    '''
    
    df['abs_elo_diff']=[row[0]-row[1] for row in df[['WhiteElo', 'BlackElo']].values]
    
    return df


def keep_rated_games(df):
    df = df[[' rated' in x for x in df.Event]]
    
    return df


def keep_time_length_games_greater_than(df, time_control_base_minimum=3600):
    df = df[df['TimeControl_Base']>=time_control_base_minimum]
    
    return df