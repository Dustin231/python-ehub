Getting Started Guide: E-hub Tool (v5.4)
============


**Test4**

*Test4*

***Test4***

Requirements
------------

- Python 3.6+
- Python libraries: pandas, numpy, Pyomo
- Solver supported by Pyomo (e.g., glpk, gurobi); note: current model tested/configured with gurobi only
- Spreadsheet editor (e.g., Microsoft Excel, OpenOffice)


Installation Quick Start
---------------

***Python:***
-	An easy was to get started with Python is by installing the Anaconda package from: https://www.anaconda.com/download/
-	Download and install the appropriate Python version 3.6+ for your machine

***Libraries:***
-	Launch Anaconda Prompt
-	Enter the following commands to install pandas, numpy, and Pyomo libraries, respectively (may require administrator access):
1.	“conda install pandas”
2.	“conda install numpy”
3.	“pip install pyomo”

***Solver:***
-	To install the glpk solver, enter the command: “conda install -c conda-forge glpk” (using Anaconda Prompt)

***OpenOffice:***
-	Download and install the appropriate OpenOffice software package for your machine from:  https://www.openoffice.org/download/ 


Defining a Model
---------------

The energy hub model is defined using a spreadsheet. The input spreadsheet template is located under the “cases” GitHub folder. Several worksheets are used to define the model. These are briefly described below. Further information regarding the input fields and their specification is provided in the comments on individual cells within the worksheets.
Note that the complete optimization problem is defined and documented in the “E-hub optimization problem.docx” file under the GitHub “docs” folder.

***Worksheet Descriptions:***

*General:* Overarching optimization problem and modeling parameters

*Energy Carriers:* Complete list of energy carriers referenced in the model

*Imports:* Imported energy carriers; note that every model requires at least one import in order to satisfy energy balance equations.

*Exports:* Exported energy carriers, if any

*Demand:* Energy carrier demand by hub

*Energy Converters:* Energy conversion technologies; at least one must be specified. Multi input/output conversion processes with fixed input and output shares may be specified. For a more complete explanation (including graphics), please refer to the *Getting Started* file under the GitHub "docs" folder.

*Storage:* Storage technologies, if any

*Network:*  Network connections, if any

*Solar Profile:* Solar irradiation profile and energy carrier, if any


Defining a Model
---------------
