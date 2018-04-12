#-----------------------------------------------------------------------------#
# Run Model - Main
#-----------------------------------------------------------------------------#

from run_mod import *

# input data file location
input_path=r'C:\Users\boa\Documents\Repositories_Github\python-ehub\cases\eawag_experiment\input_data.xlsx'

# results saved to text file location
result_file = 'C:/Users/boa/Documents/Repositories_Github/python-ehub/results/eawag_experiment/results.txt'

# model parameters saved to text file location
param_file = 'C:/Users/boa/Documents/Repositories_Github/python-ehub/results/eawag_experiment/params.txt'

# execute model
run_model(input_path, param_file, result_file) # defined in run_mod.py
