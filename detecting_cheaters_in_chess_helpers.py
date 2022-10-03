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
        for i in cols:
            if i not in df.columns:
                print(f'column {i} not found')
                return False
        return True

    def dtype_checker(df, schema):
        for i,j in schema.items():
            if i in df.columns:
                if df[i].dtype != j:
                    print(f'column {i} does not match dtype {j}')
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