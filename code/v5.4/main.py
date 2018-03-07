#-----------------------------------------------------------------------------#
# Run Model - Main
#-----------------------------------------------------------------------------#

from run_mod import *

# input data file location
input_path=r'C:\python-ehub-NextGen\cases\Ehub input - multi hub example v5.4.xlsx'

# results saved to text file location
result_file = 'C:/python-ehub-NextGen/results excel/results_ex.txt'

# model parameters saved to text file location
param_file = 'C:/python-ehub-NextGen/results excel/param_ex.txt'

# execute model
run_model(input_path, param_file, result_file) # defined in run_mod.py