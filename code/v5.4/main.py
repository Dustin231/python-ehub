#-----------------------------------------------------------------------------#
# Run Model - Main
#-----------------------------------------------------------------------------#

from run_mod import *

# input data file location
#excel_path=r'C:\Users\yam\Documents\Ehub modeling tool\python-ehub-dev\cases\Ehub input - multi-node example v5.3.xlsx'
excel_path=r'C:\Users\yam\Documents\Test cases\Test case - multi hub net micro r5.xlsx'

# results saved to text file location
#result_file = 'C:/Users/yam/Documents/GitHub/python-ehub/cases/Results/Results.txt'
result_file = 'C:/Users/yam/Documents/Test cases/Results/results_r5.txt'

# model parameters saved to text file location
param_file = 'C:/Users/yam/Documents/Test cases/Results/param_r5.txt'

# execute model
run_model(excel_path, param_file, result_file) # defined in run_mod.py