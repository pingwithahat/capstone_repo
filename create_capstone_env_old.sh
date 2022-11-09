#!/bin/bash

# NOTE: Run this as a 'source create_capstone_env.sh'

# This will create the required conda environment, $1 should be the
# name you want to give the environment (capstone_env is recommended) and
# $2 is the capstone_env.txt file and $3 is the requirements.txt file

#conda create --name "$1" --file capstone_env.txt
conda create --name "$1" --file "$2"

conda activate "$1"

#pip install chess

#pip install -r requirements.txt
pip install -r "$3"

echo "Created environment"
