#!/bin/bash

# Note: run this as 'source save_env_yml.sh'

# This script will save the specified environment used packages to 
# the specified file. 
# $1 should be the environment to be saved.
# $2 should be the conda_env.yml file name.

conda activate "$1"

conda env export --no-builds | grep -v "name\|prefix" > "$2.yml"

conda deactivate