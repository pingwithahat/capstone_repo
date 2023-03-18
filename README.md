This project aimed at investigating statistical methods of detecting chess cheaters. 

The order of notebooks goes 1 - Loading, Cleaning, EDA, Feature Engineering; 2 - Modelling and Results

`detecting_cheaters_in_chess_helpers.py` is a .py file containing functions used in the analysis notebooks. 

To recreate the conda environment, `env_creation` contains bash scripts as well as a .yml file. Please note the scripts and .yml file were only tested with Windows compatability. 

Data required is available for download from https://www.dropbox.com/sh/uqwcv9v759tzh2i/AABMYjBshvsEGGIgUgi_qfjDa?dl=0 or https://drive.google.com/file/d/1f8LfouP_WfQn6Rj9xJCQI8Cg-m09zcwc/view?usp=share_link

The python script to convert pgn to json was taken and modified from https://github.com/moritzhambach/Detecting-Cheating-in-Chess

The `notebooks_wip` contains notebooks that were used during the project. The 'while eval posit' and 'while grid search' were notebooks created to run code (i.e. while using stockfish to evaluate every position, and while performing grid searc to optimise models) in parallel to further analysis. 
