#!/bin/bash

# NOTE: Run this as a 'source create_env_yml.sh'

# This will create the required conda environment using a .yml file.
# $1 should be the environment name.
# $2 should be the env.yml file.


conda env create --name "$1" --file "$2"

echo "Script end"
