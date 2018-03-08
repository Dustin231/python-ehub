Getting Started Guide: E-hub Tool (v5.4)
============

#Test1
##Test2
###Test3

**Test4**

Requirements
------------

- Python 3.6+
- Python libraries: pandas, numpy, Pyomo
- Solver supported by Pyomo (e.g., glpk, gurobi); note: current model tested/configured with gurobi only
- Spreadsheet editor (e.g., Microsoft Excel, OpenOffice)


Installation Quick Start
---------------

Python:
-	An easy was to get started with Python is by installing the Anaconda package from: https://www.anaconda.com/download/
-	Download and install the appropriate Python version 3.6+ for your machine

Libraries:
-	Open Anaconda Prompt
-	Enter the following commands to install pandas, numpy, and Pyomo libraries, respectively (may require administrator access):
1.	“conda install pandas”
2.	“conda install numpy”
3.	“pip install pyomo”

Solver:
-	To install the glpk solver, enter the command: “conda install -c conda-forge glpk” (using Anaconda Prompt)

OpenOffice:
-	Download and install the appropriate OpenOffice software package for your machine from:  https://www.openoffice.org/download/ 

Common errors
---------------

**DisDemands Error:**

- Comment out or delete this line in the DisDemands function: `for i, value in enumerate(np.array(self.TechOutputs[[val-2]],dtype=int)):`

- Change it to this: `for i, value in enumerate(np.array(pd.Series.to_frame(self.TechOutputs.iloc[:,val-2]),dtype=int)):`
