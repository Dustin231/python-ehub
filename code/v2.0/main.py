#-----------------------------------------------------------------------------#
# Run Model - Main
#-----------------------------------------------------------------------------#

from run_mod import *

# input data file location
input_path=r'C:\Users\yam\Documents\GitHub\python-ehub\cases\demos\build v2.0\D5 - multi hub net cen micro.xlsx'

# results saved to text file location
result_file = 'C:/Users/yam/Documents/Test cases/Results/results_test.txt'

# model parameters saved to text file location
param_file = 'C:/Users/yam/Documents/Test cases/Results/param_test.txt'

# execute model
run_model(input_path, param_file, result_file) # defined in run_mod.py