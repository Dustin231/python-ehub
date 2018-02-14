
# coding: utf-8

# In[24]:


from pyomo.core import *
from pyomo.opt import SolverFactory, SolverManagerFactory
import pyomo.environ
import pandas as pd
import numpy as np
from itertools import compress
#import seaborn as sns
#import xarray as xr
#import matplotlib as mpl
#import matplotlib.pyplot as plt
from operator import add
import scipy.io as sio


# In[25]:


#%matplotlib inline #show plots in jupyter notebook and not in a new window


# In[26]:


from IPython.core.display import HTML
HTML("<style>.container { width:80% !important; }</style>")


# # Input data class

# In[27]:


class InputData:
    
    #-----------------------------------------------------------------------
    #  Initialization when class is created
    #-----------------------------------------------------------------------
    
    def __init__(self, ExcelPath, demands='Demand',solar='Solar Profile', tech='Energy Converters', stg='Storage', gen='General', EC='Energy Carriers', net = 'Network', imp = 'Imports', exp = 'Exports'):  #if no values are passed, use default ones
        self.path = ExcelPath
        self.DemandSheet = demands
        self.SolarSheet = solar
        self.TechSheet = tech
        self.GeneralSheet = gen
        self.ECSheet = EC
        self.NetSheet = net
        self.ImpSheet = imp
        self.ExpSheet = exp
        self.StgSheet = stg
                
        self.TechOutputs = None
        self.Technologies = None
        self.DemandData = None
        self.StorageData = None
        self.EC_dict = None
        self.numberofhubs = None
#        self.numberofdemands = None
        self.numbertime = None
        self.numberec = None
        self.numbertech = None
        
        self.Initialize() #call variable intialization function
        
    def Initialize(self):
        """ 
        Initialize paramaters
        """
        self.EnergyCarriers() # list first, as this builds EC_dict which is used in other functions
        self.General()
        self.ImportData()
        self.Demanddata()
        self.TechParameters()
        self.TechOutput()
        self.Storage()
        self.Network()
        
    def LookupEC(self,ec,errmsg):
        """
        Looks up and returns energy carrier IDs for list 'ec'. Generates error message 'errmsg' if lookup fails
        """
        lu = []
        for i,EC_name in enumerate(ec):
            try:
                lu += [self.EC_dict[EC_name]]
            except KeyError: # if a lookup error occurs
                print(errmsg)
                raise # reraise last exception
            
        return lu
    
    def ParseEC(self,ec):
        """
        Parses string of energy carriers in list 'ec'. Looks up and returns dictionary of corresponding energy carrier IDs
        """
        
        # lookup input energy carriers
        ecID = {} # dictionary to store input EC IDs for each technology
        for i in range(0,len(ec)):
            eci = [x.strip() for x in str(ec[i]).split(',')] # parse comma-delimited string and save as list
            lu = self.LookupEC(eci,"Technology input/output energy carrier lookup failed; check Technology spreadsheet") # lookup EC IDs
            ecID[i+1] = lu # store input ID for technology; CHANGE FIX, +1 because electricity  ID = 1  
        
        return ecID
    
    def ParseShare(self,shr):
        """
        Parses string of fixed input shares in list 'shr'. Returns a dictionary of checked and float-formatted values
        """
        
        parseShr = {}
        
        for i in range(0,len(shr)):
            shrS = [x.strip() for x in str(shr[i]).split(',')] # parse comma-delimited string and save as list
            shrF = [float(j) for j in shrS] # convert list of strings to float
            if sum(shrF) != 1 and np.isnan(sum(shrF)) == 0: # generate error if user input shares that do not sum to 1
                raise ValueError("Fixed input or output share does not sum to 1")
            parseShr[i+1] = shrF # store share for technology; CHANGE FIX, +1 because electricity  ID = 1  
            
        return parseShr
        
    def TechParameters(self):
        """
        Load from excel technologies and their parameters.
        Skip all other lines on the sheet that are not related to the technologies.
        """
        Technologies=pd.read_excel(self.path,sheetname=self.TechSheet, skiprows=2, index_col=0) #technology characteristics
        Technologies=Technologies.dropna(axis=1, how='all') #technology characteristics 
        self.instCapTech = Technologies.loc["Capacity (kW)"] # pre-installed capcities; data is the same for all hubs; need to keep NaN flag
        
        self.numbertech = Technologies.shape[1]
        
        # lookup and store input/output energy carriers
        self.inEC = self.ParseEC(Technologies.loc["Input energy carrier"])
        self.outEC = self.ParseEC(Technologies.loc["Output energy carrier"])
        
        # check and store given fixed input/output shares
        self.inShr = self.ParseShare(Technologies.loc["Fixed input share"])
        self.outShr = self.ParseShare(Technologies.loc["Fixed output share"])
        
        # check to ensure number of input/output share values matches the number of input/output energy carriers
        for i, vali in enumerate(self.inShr): # for each technology
            if len(self.inShr[vali]) > 1: # if fixed input shares are given (i.e., multiple input shares values are given for a technology)
                if len(self.inEC[vali]) != len(self.inShr[vali]): # check to ensure number of input ECs matches number of fixed input shares provided
                    raise ValueError("Mismatch between number of input energy carriers and fixed input shares; check Technology spreadsheet") # if not, raise error
            if len(self.outShr[vali]) > 1: # if fixed output shares are given (i.e., multiple output shares values are given for a technology)
                if len(self.outEC[vali]) != len(self.outShr[vali]): # check to ensure number of output ECs matches number of fixed output shares provided
                    raise ValueError("Mismatch between number of output energy carriers and fixed output shares; check Technology spreadsheet") # if not, raise error
                
        # store list of technology IDs with a fixed output share
        techfixout = []
        techfreeout = []
        for i, techi in enumerate(self.outEC): # for each technology
            if len(self.outShr[techi]) > 1: # if there is more than one output share specified
                techfixout.append(techi)
            else: # the output share is only one (i.e., either = 1 or is nan (i.e., field left blank))
                techfreeout.append(techi)
        self.techFixOut = techfixout
        self.techFreeOut = techfreeout
        
        # store list of technology IDs with a fixed input share
        techfixin = []
        for i, techi in enumerate(self.inEC): # for each technology
            if len(self.inShr[techi]) > 1: # if there is more than one input share specified
                techfixin.append(techi)
        self.techFixIn = techfixin       
        
        # store part load efficiencies and identify technologies
        self.partload = dict(zip(range(1,self.numbertech+1), np.array(Technologies.loc['Minimum load (%)'])/100)) # create dictionary with 1:N and part load efficiency values
        self.techPL = [k for (k,v) in self.partload.items() if v > 0] # select partload tech IDs where partload efficiency is greater than 0       
        
        # lookup and store solar EC
        solEC=pd.read_excel(self.path,sheetname=self.SolarSheet, skiprows=1, index_col=0)
        solEC=solEC.iloc[0][0]
        self.solECID = []
        self.techSol = []
        if type(solEC) is not float: # solEC must be specified (i.e., is a string and is a not NaN (float))
            self.solECID = self.LookupEC([solEC], "Solar energy carrier label not recognized. Check solar input spreadsheet.")
            # identify solar technologies
            techsol = []
            for (k,v) in self.inEC.items():
                if self.solECID[0] in v:
                    techsol.append(k)
            self.techSol = techsol
        
        # store solar specific power
        self.solkWm2 = dict(zip(range(1,Technologies.shape[1]+1), np.array(Technologies.loc['Solar specific power (kW/m2)']))) # create dictionary with 1:N and efficiency values
        
        # store all input technology data for each hub                
        Technologies.loc["Maximum capacity (kW)"]=Technologies.loc["Maximum capacity (kW)"].fillna(float('inf')) # replace nan values for max cap with infinity
        Technologies.loc["Maximum output (kWh)"]=Technologies.loc["Maximum output (kWh)"].fillna(float('inf')) # replace nan values for max cap with infinity
        Technologies=Technologies.fillna(0) # replace remaining nan values with zero
        
        # Important to keep following dictionaries after the preceding command (fillna(0)); i.e., values that have not been specified are set to zero
        
        # store efficiencies
        self.eff = dict(zip(range(1,self.numbertech+1), np.array(Technologies.loc['Efficiency (%)'])/100)) # create dictionary with 1:N and efficiency values
        
        # store investment cost
        self.techInvCost = dict(zip(range(1,self.numbertech+1), np.array(Technologies.loc["Investment cost (CHF/kW)"])))
        
        # store CO2 factors
        self.techCO2 = dict(zip(range(1,self.numbertech+1), np.array(Technologies.loc["CO2 investment (kg-CO2/kW)"])))
        
        # store O&M variable and fixed
        OMV = np.array(Technologies.loc["Variable O&M cost (CHF/kWh)"])
        OMF = np.array(Technologies.loc["Fixed O&M cost (CHF/kW)"])
        OMV[pd.isnull(OMV)] = 0 # replace blanks (nan) with 0
        OMF[pd.isnull(OMF)] = 0 # replace blanks (nan) with 0
        self.OMvar = dict(zip(range(1,self.numbertech+1), OMV))
        self.OMfix = dict(zip(range(1,self.numbertech+1), OMF))
        
        dd={}
        for i in range(self.numberofhubs):
            dd[i]=Technologies
        self.Technologies=pd.Panel(dd) #each hub has it's own technology dataframe
    
    def TechOutput(self):
        """
        Load from excel output of technologies.
        Skip all other lines on the sheet that are not related to the technologies output.
        """
        TechOutputs=pd.read_excel(self.path,sheetname=self.TechSheet, skiprows=18, index_col=0, skip_footer=31) #Output matrix
        
        TechOutputs=TechOutputs.dropna(axis=0,how='all')  #Output matrix
        TechOutputs=TechOutputs.dropna(axis=1,how='all')  #Output matrix
        self.TechOutputs=TechOutputs
        
    def ImportData(self):
        """
        Retrieve import EC data from Excel "Imports" spreadsheet
        """
        Imp=pd.read_excel(self.path,sheetname=self.ImpSheet, skiprows=2, index_col=0)
        Imp=Imp.dropna(axis=0,how='all')
        
        Imp.loc[:,"Maximum supply (kWh)"] = Imp.loc[:,"Maximum supply (kWh)"].fillna(float('inf')) # replace nan values for max cap with infinity
        Imp=Imp.fillna(0) # fill the remaining blanks (nan) with zero values
        
        # lookup EC IDs
        ecID = self.LookupEC(Imp.index.tolist(), "Supply energy carrier lookup error")
        Imp.index = ecID
        
        self.ImportData = Imp
        
    
    def Demanddata(self):
        """
        Retrieve network data from Excel "Demand Data" spreadsheet
        """
        DemandDatas=pd.read_excel(self.path,sheetname=self.DemandSheet, header=None, skiprows=1)
#        self.numberofdemands = DemandDatas[1][0]
        self.DemandData = DemandDatas.loc[4:DemandDatas.shape[0],1:DemandDatas.shape[1]] # demand data; extract data from index row label 3:last row, and column label 1:last column
        self.DemandHub = DemandDatas.loc[2,1:DemandDatas.shape[1]].values.tolist() # hub data; convert to list to be consistent with DemEC formatting
        self.numbertime= self.DemandData.shape[0]
        dec = DemandDatas.loc[1,1:DemandDatas.shape[1]] # row of demand energy carriers
        
        # lookup energy carrier names
        DemEC = self.LookupEC(dec,"Energy carrier lookup failed in Demand Data input spreadsheet")              
        self.DemandEC = DemEC # demand EC data
        
    def Storage(self):
        """
        Load from excel storage data.
        Skip all other lines on the sheet that are not related to the storage.
        """
        Storage=pd.read_excel(self.path,sheetname=self.StgSheet, skiprows=2, index_col=0)
        Storage=Storage.dropna(axis=1, how='all')
        self.instCapStg = Storage.loc["Capacity (kWh)"] # pre-installed capcities; data is the same for all hubs; need to keep NaN flag
        Storage.loc["Maximum capacity (kWh)"] = Storage.loc["Maximum capacity (kWh)"].fillna(float('inf')) # replace nan values for max cap with infinity
        Storage=Storage.fillna(0) # fill the remaining blanks (nan) with zero values
        self.StorageData=Storage
        
        # Convert stored energy carrier input data to assigned energy carrier ID in dictionary, save in list
        StgE = self.LookupEC(self.StorageData.loc["Stored energy carrier"],"Energy carrier lookup failed for storage data")
        self.stgE = StgE
        
    def General(self):
        """
        Get number of hubs, interest rate, objective minimization target, and max CO2 emissions from input spreadsheet
        """    
        gendata=pd.read_excel(self.path,sheetname=self.GeneralSheet, skiprows=1, index_col=0) #Output matrix
        gendata=gendata.dropna(axis=1,how='all')  #Output matrix
        self.numberofhubs=int(gendata.loc["Number of hubs"][0])
        self.interest=gendata.loc["Interest rate (%)"][0]/100
        self.minobj=gendata.loc["Minimization objective"][0]
        self.maxcarbon=gendata.loc["Maximum CO2 emissions (kg-CO2)"][0]
        self.bigm=gendata.loc["Big M"][0]
            
    def EnergyCarriers(self):
        """
        Get the list of energy carriers used in the model, assign the energy carriers values from 1 to N in a dictionary, and save the energy carrier values in a list
        """
        
        # Read data
        ECs = pd.read_excel(self.path,sheetname=self.ECSheet, header=None, skiprows=3)
        ECs = ECs.dropna(axis=1, how='all')
        
        self.numberec = ECs.shape[0]
        
        # Create dictionary
        dic = {}
        for i in range(self.numberec):
            dic[ECs.loc[i][0]] = i+1
        self.EC_dict = dic
        
    def Network(self):
        """
        Retrieve network data from Excel "Network" spreadsheet
        """
        
        # Read data
        net = pd.read_excel(self.path,sheetname=self.NetSheet, header=None, skiprows=2, index_col = 0)
        net = net.dropna(axis=1, how='all')
        self.NetworkData = net
        flag = not net.empty # boolean flag for network data; ==  0 if no network exists, == 1 if network exists
        
        if flag is True: # check if network link data is given
            # Convert stored energy carrier input data to assigned energy carrier ID in dictionary, save in list
            NetE = self.LookupEC(self.NetworkData.loc["Energy carrier"],"Energy carrier lookup failed for network link under Network input spreadsheet")             
            self.netE = NetE
            
            # Store other parameters
            self.node1 = self.NetworkData.loc["Node 1"]
            self.node2 = self.NetworkData.loc["Node 2"]
            self.netLength = self.NetworkData.loc["Length (m)"]
            self.netLength = self.netLength.fillna(0)
            self.netLoss = self.NetworkData.loc["Network loss (fraction/m)"]
            self.netLoss = self.netLoss.fillna(0)
            self.instCap = self.NetworkData.loc["Installed capacity (kW)"]
            self.instCap = self.instCap.fillna(0) # CHANGE, leave as NAN
            self.maxCap = self.NetworkData.loc["Maximum capacity (kW)"]
            self.maxCap = self.maxCap.fillna(float('inf')) # if maximum is not given, assign it to infinity
            self.minCap = self.NetworkData.loc["Minimum capacity (kW)"]
            self.minCap = self.minCap.fillna(0)
            self.invCost = self.NetworkData.loc["Investment cost (CHF/kW/m)"]
            self.invCost = self.invCost.fillna(0)
            self.OMVCost = self.NetworkData.loc["Variable O&M cost (CHF/kWh)"]
            self.OMVCost = self.OMVCost.fillna(0)        
            self.OMFCost = self.NetworkData.loc["Fixed O&M cost (CHF/kW)"]
            self.OMFCost = self.OMFCost.fillna(0)
            self.netCO2 = self.NetworkData.loc["CO2 investment (kg-CO2/kW/m)"]
            self.netCO2 = self.netCO2.fillna(0)
            self.netlife = self.NetworkData.loc["Lifetime (years)"]
            self.uniFlow = self.NetworkData.loc["Uni-directional flow? (Y)"]
            self.uniFlow = self.uniFlow.str.lower() # convert to lower case

    
    #------------------------------------------------------------------------------------
    #  Functions used for translating dataframe/panels to dictionary format used by Pyomo
    #-----------------------------------------------------------------------------------

    def Dict1D(self,dictVar,dataframe):
        """
        When the key in Pyomo dict is 1-D and it's equal to the order of data.
        """
        for i,vali in enumerate(dataframe.index):
            dictVar[i+1]=round(dataframe.iloc[i][1],4)
        return dictVar

    def Dict1D_val_index(self,dictVar,dataframe):
        """
        When the key in Pyomo dict is 1-D and it's equal to the value of dataframe index name.
        """
        for i,vali in enumerate(dataframe.index):
            dictVar[vali]=round(dataframe.iloc[i][1],4)
        return dictVar

    def DictND(self,dictVar,dataframe):
        """
        When the key in Pyomo dict is 2-D and it's equal to the value of dataframe index/column name.
        """
        for i,vali in enumerate(dataframe.index):
            for j,valj in enumerate(dataframe.columns):
                dictVar[vali,valj]=dataframe.loc[vali][valj]
        return dictVar
    
    def DictPanel(self, dictVar,panel):
        """
        When the key in Pyomo dict is 3-D+ and it's equal to the order of data.
        """
        for x,valx in enumerate(panel.items):
            for i, vali in enumerate(panel[valx].dropna(axis=0, how ='all').index):
                for j, valj in enumerate(panel[valx].dropna(axis=1, how ='all').columns):
                    dictVar[x+1,j+1, i+1] = panel[valx].loc[vali][valj] #Pyomo starts from 1 and Python from 0
        return dictVar
    
    
    #----------------------------------------------------------------------
    # Load profiles
    #----------------------------------------------------------------------
    
    def Demands(self):
        """
        Return Pyomo formatted demand data for all hubs.
        """    
        
        # create a dictionary of zeros which will be updated with demand values
        dummy={}
        for i in range(self.numberofhubs):
            dummy[i]=np.zeros((self.numberec,self.numbertime))
            
        # update demand data dictionary one column at a time
        for i in range(0,self.DemandData.shape[1]):
            hub = self.DemandHub[i]
            ec = self.DemandEC[i]
            dummy[hub-1][ec-1, :] = self.DemandData.iloc[:,i]

        Demand=pd.Panel(dummy)
        loads_init={}
        loads_init=self.DictPanel(loads_init,Demand)
        return loads_init
    
    
    
    
    #----------------------------------------------------------------------
    # Irradiation
    #----------------------------------------------------------------------
    
    def SolarData(self):
        """
        Return Pyomo formatted solar data.
        """
        solar_init={}
        SolarData=pd.read_excel(self.path,sheetname=self.SolarSheet, skiprows=4)
        if SolarData.empty is False:
            SolarData = SolarData.loc[0:SolarData.shape[0],'Irradiation (kW/m2)'] 
            SolarData= SolarData.dropna(axis=0,how='all')
            SolarData = pd.DataFrame(SolarData)
            SolarData.columns=[1]
            solar_init=self.Dict1D(solar_init,SolarData)
        return solar_init
    
    
    
    
    #----------------------------------------------------------------------
    # Dispatch technologies
    #----------------------------------------------------------------------
    
#    def CHP_list(self):
#        """
#        Return Pyomo formatted dispatch tech list for dispatch set.
#        """  
#        Dispatch_tech=pd.DataFrame(self.TechOutputs.sum(0)) #find dispatch tech (e.g. CHP)
#        CHP_setlist=[]
#        for n,val in enumerate(Dispatch_tech[0]):
#            if val>1:
#                CHP_setlist.append(n+1) #first is electricity, +1 since it starts at 0
#        return CHP_setlist
    
    
    
    
    #----------------------------------------------------------------------
    # Roof technologies - MOVE TO CUSTOM CONSTRAINT MODULE, user should manually indicate which techs are roof top (e.g., PV panel could be for ground)
    #----------------------------------------------------------------------
    
    def Roof_tech(self):
        """
        Return Pyomo formatted roof tech (e.g. PV, ST) for roof tech set. - MOVE TO CUSTOM CONSTRAINT MODULE, user should manually indicate which techs are roof top (e.g., PV panel could be for ground)
        """  
        Roof_techset=[]
        for n,val in enumerate(self.Technologies[0].loc["Solar specific power (kW/m2)"]): #take only 0th item since it is the same for all hubs
            if val>0:
                Roof_techset.append(n+1)
                
        return Roof_techset
   
    
    #----------------------------------------------------------------------
    # C-matrix
    #----------------------------------------------------------------------
        
#    def cMatrix(self):
#        """
#        Return Pyomo formatted C-matrix data.
#        """
#        
#        
#        #Based on the + - values, prepare data for generating coupling matrix        
#        TechOutputs2=self.TechOutputs.multiply(np.array(self.Technologies[0].loc['Efficiency (%)']))
#        TechOutputs2.loc[TechOutputs2.index!='Electricity']=TechOutputs2.loc[(TechOutputs2.index!='Electricity')].multiply(np.array(self.Technologies[0].loc['Fixed output share'].fillna(value=1).replace(0,1))) #multiply all positive values of output matrix by COP/HER/efficiency
#        TechOutputs2[TechOutputs2<0]=TechOutputs2[TechOutputs2<0].divide(np.array(self.Technologies[0].loc['Efficiency (%)'].fillna(value=1).replace(0,1))) #Get original value for negative values
#        #TechOutputs2[TechOutputs2<0]=-1 #this is only needed if there is single input for e.g. HP
#
#        addGrid=np.zeros(self.numberofdemands,) #electricity is always present and added on the first place
#        addGrid[0]=1 #add electricity to coupling matrix
#        Grid=pd.DataFrame(addGrid,columns=["Grid"],index=range(1,self.numberofdemands+1)).transpose() # FIX
#        
#        Cmatrix=TechOutputs2.transpose()
#        Cmatrix.columns = list(range(1,len(Cmatrix.columns)+1))
#        Cmatrix=pd.concat([Grid,Cmatrix]) #combine grid with cmatrix containing only technologies
#        Cmatrix.index=list(range(1,len(TechOutputs2.columns)+2)) #format column/row names for proper translation to Pyomo dict format
#        Cmatrix.columns=list(range(1,len(TechOutputs2.index)+1)) #format column/row names for proper translation to Pyomo dict format
#        cMatrixDict={}
#        cMatrixDict=self.DictND(cMatrixDict,Cmatrix)
#        
#        return cMatrixDict
    
    #----------------------------------------------------------------------
    # Efficiencies
    #----------------------------------------------------------------------
        
    def EffInit(self):
        """
        Return Pyomo formatted efficiencies for technologies without a fixed output share
        """

        d = {}
        for i in self.techFreeOut:
            d[i] = self.eff[i]
            
        return d
            
        
    def OutFixInit(self):
        """
        Return Pyomo formatted efficiencies for technologies with a fixed output share
        """
        
        d= {}
        for i in self.techFixOut: # for each technology
            for j in range(1,self.numberec+1): # for each energy carrier
                if j in self.outEC[i]:
                    d[i,j] = self.eff[i] * self.outShr[i][self.outEC[i].index(j)] / self.outShr[i][0]
                else:
                    d[i,j] = 0
                    
        return d
    
    #----------------------------------------------------------------------
    # Input shares
    #----------------------------------------------------------------------
    
    def InFixInit(self):
        """
        Return Pyomo formatted values for technologies with fixed input shares
        """
    
        d= {}
        for i in self.techFixIn: # for each technology
            for j in range(1,self.numberec+1): # for each energy carrier
                if j in self.inEC[i]:
                    d[i,j] = self.inShr[i][self.inEC[i].index(j)]
                else:
                    d[i,j] = 0
                    
        return d
    
    
    #----------------------------------------------------------------------
    # Part load information
    #----------------------------------------------------------------------
    
    # CHANGE: No longer used; delete
#    def PartLoad(self):
#        """
#        Return Pyomo formatted information about technologies' part load.
#        """
#        PartLoad=self.Technologies[0].loc["MinLoad (%)",]/100
#
#        partload=self.TechOutputs.iloc[0:1].mul(list(PartLoad),axis=1)
#        partload=pd.concat([partload,self.TechOutputs.iloc[1:].mul(list(PartLoad),axis=1)], axis=0)
#        partload=partload.abs()
#        partload=partload.transpose()
#        partload.index=list(range(1,len(self.TechOutputs.columns)+1)) # CHANGE elec == 1
#        partload.columns=list(range(1,len(self.TechOutputs.index)+1))
#        SolartechsSets=list(compress(list(range(1,len(self.Technologies[0].columns)+1)), list(self.Technologies[0].loc["Area (m2)"]>0))) # CHANGE elec == 1
#
#        for i in SolartechsSets:
#            partload.drop(i, inplace=True) #remove from the part load list roof tech
#
#        PartloadInput={}
#        PartloadInput=self.DictND(PartloadInput,partload)
#        
#        return PartloadInput
    
    
    
    
    #----------------------------------------------------------------------
    # Maximum capacities
    #----------------------------------------------------------------------
    
#    def MaxCapacity(self): # replaced with def TechLimitInit
#        """
#        Return Pyomo formatted max capacity that can be installed for all technologies present in specific hub for all hubs.
#        """
#        maxCap={}
#        for n in range(self.numberofhubs):
#            MaxCap=pd.DataFrame(self.Technologies[n].loc["Maximum capacity (kW)",])
#            MaxCap.index=list(range(1,self.numbertech+1))
#            maxCap[n]= MaxCap
#
#            if self.numberofhubs>2:
#                for k in maxCap[n].index:
#                    if isinstance(self.Technologies[n].loc["Hubs"][k-1],float) or isinstance(self.Technologies[n].loc["Hubs"][k-1],int):#check if technology is present in one hub
#                        if int(n+1) != int(self.Technologies[n].loc["Hubs"][k-1]):
#                            maxCap[n].loc[k] = 0 #if tech is not present in n-th hub installed cap has to be 0
#                    else:
#                        if str(n+1) not in [x.strip() for x in self.Technologies[n].loc["Hubs"][k-1].split(',')]: #split at comma # CHANGE elec == 1
#                            maxCap[n].loc[k] = 0 #if tech is not present in n-th hub installed cap has to be 0
#                            
#            elif self.numberofhubs==2: #check if technology is present in one or more hubs, otherwise there is error
#                for k in maxCap[n].index:
#                    if isinstance(self.Technologies[n].loc["Hubs"][k-1],float) or isinstance(self.Technologies[n].loc["Hubs"][k-1],int): # CHANGE elec == 1
#                        if int(n+1) != int(self.Technologies[n].loc["Hubs"][k-1]): # CHANGE elec == 1
#                            maxCap[n].loc[k] = 0 #if tech is not present in n-th hub installed cap has to be 0 
#        #format column/row name
#        CapacitiesPanel=pd.Panel(maxCap) #create panel for dict
#        
#        Capacities = CapacitiesPanel.to_frame()
#        Capacities.reset_index(inplace = True)
#        Capacities.index = Capacities['major']
#        del Capacities['major']
#        del Capacities['minor']
#        Capacities.columns = [int(x)+1 for x in Capacities.columns] #format column/row names for proper translation to Pyomo dict
#        del Capacities.index.name
#        Capacities = Capacities.transpose()
#        
#        maximalcapacities={}
#        maximalcapacities = self.DictND(maximalcapacities, Capacities)
#        return maximalcapacities
    
    def TechLimitInit(self, lookup):
        """
        Return Pyomo formatted dict of limit per tech and hub
        """

        dlim = np.zeros((self.numberofhubs, self.numbertech)) 
        limit = pd.DataFrame(list(data.Technologies[0].loc[lookup]))
                       
        for i in range(self.numbertech):
            for j in range(self.numberofhubs):
                #check if technology is present in each hub
                if isinstance(self.Technologies[0].loc["Hubs"][i],float) or isinstance(self.Technologies[0].loc["Hubs"][i],int):  # present in only one hub
                    dlim[j,i] = limit.iloc[i][0]
                        
                elif str(j+1) in list(self.Technologies[0].loc["Hubs"][i]): # if hub j exists for tech i (in a list of multiple hubs)
                    dlim[j,i] = limit.iloc[i][0]

        lim = pd.DataFrame(dlim)
        lim.index = lim.index+1
        lim.columns = lim.columns+1
        limit_dict={}
        limit_dict = self.DictND(limit_dict, lim)

        return limit_dict

    
#    def MaxCapacity(self): - DELETE
#        """
#        Return Pyomo formatted max capacity that can be installed for non-roof technologies present in specific hub for all hubs.
#        """
#        maxCap={}
#        for n in range(self.numberofhubs):
#            MaxCap=pd.DataFrame(self.Technologies[n].loc["Maximum Capacity",])
#            MaxCap.index=list(range(1,len(self.Technologies[n].loc["Maximum Capacity",].index)+1)) # CHANGE elec == 1
#            maxCap[n]= MaxCap
#
#            SolartechsSets=list(compress(list(range(1,len(self.Technologies[n].columns)+1)), list(self.Technologies[n].loc["Area (m2)"]>0))) # CHANGE elec == 1
#            maxCap[n] = maxCap[n].drop(SolartechsSets) #if it is roof tech, remove it from dict
#            if self.numberofhubs>2:
#                for k in maxCap[n].index:
#                    if isinstance(self.Technologies[n].loc["Hubs"][k-1],float) or isinstance(self.Technologies[n].loc["Hubs"][k-1],int)or isinstance(self.Technologies[n].loc["Hubs"][k-1],int): #check if technology is present in one or more hubs, otherwise there is error # CHANGE elec == 1
#                        if int(n+1) != int(self.Technologies[n].loc["Hubs"][k-1]):# # CHANGE elec == 1
#                            maxCap[n].loc[k] = 0 #if tech is not present in n-th hub installed cap has to be 0
#                    else:
#                        if str(n+1) not in [x.strip() for x in self.Technologies[n].loc["Hubs"][k-1].split(',')]:#if>=elif # CHANGE elec == 1
#                            maxCap[n].loc[k] = 0 #if tech is not present in n-th hub installed cap has to be 0
#            elif self.numberofhubs==2:
#                for k in maxCap[n].index:
#                    if isinstance(self.Technologies[n].loc["Hubs"][k-1],float): #check if technology is present in one or more hubs, otherwise there is error # CHANGE elec == 1
#                        if int(n+1) != int(self.Technologies[n].loc["Hubs"][k-1]): # CHANGE elec == 1
#                            maxCap[n].loc[k] = 0 #if tech is not present in n-th hub installed cap has to be 0 
#        #format column/row name                
#        CapacitiesPanel=pd.Panel(maxCap)
#        
#        Capacities = CapacitiesPanel.to_frame() #create panel for dict
#        Capacities.reset_index(inplace = True)
#        Capacities.index = Capacities['major']
#        del Capacities['major']
#        del Capacities['minor']
#        Capacities.columns = [int(x)+1 for x in Capacities.columns] #format column/row names for proper translation to Pyomo dict
#        del Capacities.index.name
#        Capacities = Capacities.transpose()
#        
#        maximalcapacities={}
#        maximalcapacities = self.DictND(maximalcapacities, Capacities)
#        return maximalcapacities
    
    
    
    def TechCapInit(self, model):
        """
        Initialize existing technology capacities; return pyomo formatted dictionary for binary cost indicator
        """
        
        YN_techcost_dict = {} # indicator to apply costs based on capacity (investment); model does not apply these costs to a pre-installed capacity
        setvals = [] # for tracking purposes only
        
        for i in range(len(self.instCapTech)): # for each tech
            techID = i+1
            if(np.isnan(self.instCapTech[i])): # if capacity is not given
                YN_techcost_dict[techID] = 1 # investment costs apply
            else:
                YN_techcost_dict[techID] = 0 # capacity is pre-defined; investment costs do not apply
            
                for j in range(self.numberofhubs): # for each hub
                    hubID = j+1
                    hublist = self.Technologies[j].loc["Hubs"][i]
                    if isinstance(hublist,float): # if only a single hub was given and it is read as float, convert to int
                        hublist = int(hublist)
                    hublist = str(hublist).split(',') # convert to string (int case); if more thane one hub is given, split hubs into a list using ',' as a delimiter
                    
                    if str(hubID) in hublist: # if capacity is given and technology is present in hub
                        capVal = self.instCapTech[i]
                        model.CapTech[hubID, techID].fix(capVal)
                        setvals.append([hubID, techID, capVal])
                            
        return (YN_techcost_dict)
    
    
    
    #----------------------------------------------------------------------
    # Predefined network connections
    #----------------------------------------------------------------------
        
     
#    def FixedNetworks(self):
#        """
#        Return Pyomo formatted pre-installed (fixed) network(s).
#        """
#        Network=pd.read_excel(self.path,sheetname="Network", index_col=0, header=None)
#        Network = Network.rename_axis(None)
#        if Network.empty!=True:
#            dummy={}
#            numberofhubs = self.numberofhubs
#            for i in range(1,self.numberofdemands+1): #Python starts from 0, Pyomo from 1
#                if isinstance(Network.loc["Demand"][1],float): #if network is only for one demand
#                    if i == int(Network.loc["Demand"][1]): #if it is for the right demand where network is present
#                        dummynetwork = np.zeros((numberofhubs,numberofhubs))
#                        for j in range(1, len(Network.columns)+1):
#                            dummynetwork[int(Network.loc["Node 2"][j]-1),int(Network.loc["Node 1"][j]-1)] = 1 #connection is present between nodes i and j
#                            dummy[i-1] = pd.DataFrame(dummynetwork)
#
#                    else:
#                        dummy[i-1] = pd.DataFrame(np.zeros((numberofhubs,numberofhubs)))
#
#                else: #if network is for multiple demands
#                    if str(i) not in list(Network.loc["Demand"][1]):
#                        dummy[i-1] = pd.DataFrame(np.zeros((numberofhubs,numberofhubs)))
#
#                    else:
#                        dummynetwork = np.zeros((numberofhubs,numberofhubs))
#                        for j in range(1, len(Network.columns)+1):
#                                dummynetwork[int(Network.loc["Node 2"][j]-1),int(Network.loc["Node 1"][j]-1)] = 1
#                                dummy[i-1] = pd.DataFrame(dummynetwork)
#
#            network = pd.Panel(dummy)
#            network_init={}
#            network_init=self.DictPanel(network_init,network)
#            return network_init
    
    
    #----------------------------------------------------------------------
    # Pyomo sets
    #----------------------------------------------------------------------
    
#    def SolarSet(self):
#        """
#        Return Pyomo formatted list for set containing only roof techs (e.g. PV, ST).
#        """
#        return list(compress(list(range(1,len(self.Technologies[0].columns)+1)), list(self.Technologies[0].loc["Solar specific power (kW/m2)"]>0)))
    
#    def SolInit(self):
#        """
#        Return ID of solar EC.
#        """
#
#        solEC=pd.read_excel(self.path,sheetname=self.SolarSheet, skiprows=1, index_col=0)
#        solEC=solEC.iloc[0][0]
#        solID = self.LookupEC([solEC], "Solar energy carrier label not recognized. Check solar input spreadsheet.")
#        
#        return solID
#    
#    def DispTechsSet(self):
#        """
#        Return Pyomo formatted list for set containing dispatch techs (e.g. CHP).
#        """
#        return list(compress(list(range(1,len(self.Technologies[0].columns)+1)), list(self.Technologies[0].loc["Area (m2)"]==0))) # CHANGE elec == 1

    # REMOVE
#    def partloadtechs(self):
#        """
#        Return Pyomo formatted list for set used for enforcing part load.
#        """
#        # CHANGE - simplify. This is simply set of techs with self.partload > 0 and not part of the solartech set
#        # CHANGE to include solartechs when solar cap is no longer in m2
#        
#        PartLoad=self.Technologies[0].loc["Minimum load (%)",]/100
#
#        partload=self.TechOutputs.iloc[0:1].mul(list(PartLoad),axis=1)
#        partload=pd.concat([partload,self.TechOutputs.iloc[1:].mul(list(PartLoad),axis=1)], axis=0)
#        partload=partload.abs()
#        partload=partload.transpose()
#        partload.index=list(range(1,len(self.TechOutputs.columns)+1))
#        partload.columns=list(range(1,len(self.TechOutputs.index)+1))
#        SolartechsSets=list(compress(list(range(1,len(self.Technologies[0].columns)+1)), list(self.Technologies[0].loc["Area (m2)"]>0)))
#
#        for i in SolartechsSets:
#            partload.drop(i, inplace=True)
#            
#        return list(partload.loc[partload.sum(axis=1)>0].index)    
    
    def EImpInit(self):
        """
        Return Pyomo formatted list for set of energy carriers which can be imported to model (supply energy carriers)
        """
        
        ec=pd.read_excel(self.path,sheetname=self.ImpSheet, skiprows=2, index_col=0)
        ec=ec.dropna(axis=0,how='all')
        ecID = self.LookupEC(ec.index.tolist(), "Import energy carrier lookup error")
        
        return ecID
    
    def EExpInit(self):
        """
        Return Pyomo formatted list for set of energy carriers which can be exported by model
        """
        
        ec=pd.read_excel(self.path,sheetname=self.ExpSheet, skiprows=2, index_col=0) #
        ec=ec.dropna(axis=0,how='all')
        
        ecID = self.LookupEC(ec.index.tolist(), "Export energy carrier lookup error")
        
        return ecID
        
    
    #----------------------------------------------------------------------
    # Find which is the primary input for capacity 
    #----------------------------------------------------------------------
#    def DisDemands(self):
#        """
#        Find which two outputs CHP is producing and save it into matrix.
#        """
#        
#        CHPlist=self.CHP_list()
#        dispatch_demands=np.zeros((len(CHPlist), 2), dtype=int)
#
#        for n,val in enumerate(CHPlist):
#            counter=0
#            #for i, value in enumerate(np.array(self.TechOutputs[[val-2]],dtype=int)):
#            for i, value in enumerate(np.array(pd.Series.to_frame(self.TechOutputs.iloc[:,val-2]),dtype=int)):
#                if value[0]>0 and counter==0:
#                    dispatch_demands[n,0]=i+1
#                    counter=1
#                if value[0]>0 and counter==1:
#                    dispatch_demands[n,1]=i+1
#
#        return dispatch_demands    
    
    
    
    #----------------------------------------------------------------------
    # Interest rate
    #----------------------------------------------------------------------
    
#    def InterestRate(self):
#        """
#        Return interest rate by reading excel.
#        """
#        Interest_rate=pd.read_excel(self.path,sheetname=self.GeneralSheet, skiprows=1, index_col=0)     
#        Interest_rate=Interest_rate.dropna(axis=1,how='all')
#        Interest_rate_R=Interest_rate.loc["Interest rate (%)"][0]/100
#        return Interest_rate_R
    
    
    
    
    #----------------------------------------------------------------------
    # Assumed lifetime of technologies
    #----------------------------------------------------------------------
    
    def LifeTime(self):
        """
        Return Pyomo formatted lifetime in years per technology.
        """

        Life=pd.DataFrame(list(self.Technologies[0].loc["Lifetime (years)"]))
        Life.columns=[1]
        Life.index=list(range(1,self.Technologies[0].shape[1]+1)) # CHANGE elec == 1
        
        lifeTimeTechs={}
        lifeTimeTechs=self.Dict1D_val_index(lifeTimeTechs, Life)
        return lifeTimeTechs
        
    
    
    
    #----------------------------------------------------------------------
    # Captial recovery factor
    #----------------------------------------------------------------------
        
    def TechCRF(self):
        """
        Return Pyomo formatted CRF values.
        """
        Life=pd.DataFrame(list(self.Technologies[0].loc["Lifetime (years)"]))
        Life.columns=[1]
        Life.index=list(range(1,self.numbertech+1))
        
        CRF=1 / (((1 + self.interest) ** Life - 1) / (self.interest * ((1 + self.interest) ** Life)))
        CRFtech={}
        CRFtech=self.Dict1D_val_index(CRFtech,CRF)
        return CRFtech
        
    
    
    
    #----------------------------------------------------------------------
    # Variable O&M costs
    #----------------------------------------------------------------------
    
#    def VarMaintCost(self):
#        """
#        Return Pyomo formatted variable O&M cost.
#        """
#        dummy = {}
#        for n in range(self.numberofhubs):
#            VarOMF=pd.DataFrame(list(self.Technologies[n].loc["O&M variable cost (CHF/kWh)"]))
#            VarOMF.columns=[1]
#            VarOMF.index=list(range(1,len(self.TechOutputs.columns)+1)) # CHANGE elec == 1, removed setting first column cost to zero
#            dummy[n] = VarOMF
#            
#        VarOMF = pd.Panel(dummy)
#        
#        VarOMF= VarOMF.to_frame() #same for all hubs
#        VarOMF.reset_index(inplace = True)
#        VarOMF.index = VarOMF['major']
#        del VarOMF['major']
#        del VarOMF['minor']
#        VarOMF.columns = [int(x)+1 for x in VarOMF.columns]
#        del VarOMF.index.name
#        VarOMF = VarOMF.transpose()
#        
#        omvCosts={}
#        omvCosts=self.DictND(omvCosts, VarOMF)
#        return omvCosts
        
    
    
    
    #----------------------------------------------------------------------
    # Lookup import EC data
    #----------------------------------------------------------------------
    
    def ImpLookup(self, lookup):
        """
        Return Pyomo formatted list for carbon factor per energy carrier.
        """

        lu = self.ImportData.loc[:,lookup]
        lu_dict = lu.to_dict()
        
        return lu_dict
       
    
    
    
    #----------------------------------------------------------------------
         
    
    
    
    #----------------------------------------------------------------------
    # Export price
    #----------------------------------------------------------------------
    
    def ExpPriceInit(self):
        """
        Return Pyomo formatted list export price per energy carrier.
        """
        Tariff=pd.read_excel(self.path,sheetname=self.ExpSheet, skiprows=2, index_col=0) #
        Tariff=Tariff.dropna(axis=0,how='all')
        Tariff=Tariff.dropna(axis=1,how='all')
        exp = Tariff["Export Price (CHF/kWh)"]
        
        ecID = self.LookupEC(exp.index.tolist(), "Export energy carrier lookup error")
        exp.index = ecID
        
        expdict = exp.to_dict()

        return expdict
    
    
    #----------------------------------------------------------------------
    # Storage 
    #----------------------------------------------------------------------
    
    def StorageCh(self):
        """
        Return Pyomo formatted list for maximum charging rate (%) per storage.
        """
        maxStorCh={}
        MaxCharge=pd.DataFrame(list(self.StorageData.loc["Maximum charging rate (%)"]/100))

        if MaxCharge.empty is False:
            MaxCharge.columns=[1]
            maxStorCh=self.Dict1D(maxStorCh, MaxCharge)
        
        return maxStorCh
    
    
    def StorageDisch(self):
        """
        Return Pyomo formatted list for maximum discharging rate (%) per storage.
        """
        maxStorDisch={}
        MaxDischarge=pd.DataFrame(list(self.StorageData.loc["Maximum discharging rate (%)"]/100))        
        if MaxDischarge.empty is False:
            MaxDischarge.columns=[1]
            maxStorDisch=self.Dict1D(maxStorDisch, MaxDischarge)        
        return maxStorDisch
    
    
    def StorageLoss(self):
        """
        Return Pyomo formatted list of self-discharge (%) per storage.
        """
        standbyLoss={}
        losses=pd.DataFrame(list(self.StorageData.loc["Standby loss (%/hour)"]/100))       
        if losses.empty is False:
            losses.columns=[1]
            standbyLoss=self.Dict1D(standbyLoss, losses)
        return standbyLoss
    
    
    def StorageEfCh(self):
        """
        Return Pyomo formatted list of charging efficiency (%) per storage.
        """
        chargingEff={}
        Ch_eff=pd.DataFrame(list(self.StorageData.loc["Charging efficiency (%)"]/100))        
        if Ch_eff.empty is False:
            Ch_eff.columns=[1]
            chargingEff=self.Dict1D(chargingEff, Ch_eff)
        return chargingEff
    
    def StorageEfDisch(self):
        """
        Return Pyomo formatted list of discharging efficiency (%) per storage.
        """
        dischargingEff={}
        Disch_eff=pd.DataFrame(list(self.StorageData.loc["Discharging efficiency (%)"]/100))
        if Disch_eff.empty is False:
            Disch_eff.columns=[1]
            dischargingEff=self.Dict1D(dischargingEff, Disch_eff)
        return dischargingEff
    
    
    def StorageMinSoC(self):
        """
        Return Pyomo formatted list of minimum state of charge (%) per storage.
        """
        minSoC={}
        min_state=pd.DataFrame(list(self.StorageData.loc["Minimum SoC (%)"]/100))
        if min_state.empty is False:
            min_state.columns=[1]
            minSoC=self.Dict1D(minSoC, min_state)
        return minSoC
    
    
    def StorageLife(self):
        """
        Return Pyomo formatted list of lifetime expectancy (yr) per storage.
        """
        lifeTimeStorages={}
        LifeBattery=pd.DataFrame(list(self.StorageData.loc["Lifetime (years)"]))
        if LifeBattery.empty is False:
            LifeBattery.columns=[1]
            LifeBattery.index=list(range(1,self.StorageData.shape[1]+1))
            lifeTimeStorages=self.Dict1D_val_index(lifeTimeStorages, LifeBattery)
        return lifeTimeStorages
    
    
    def StorageLinCost(self):
        """
        Return Pyomo formatted list of linear investment cost per storage.
        """
        linStorageCosts={}
        LinearCostStorage=pd.DataFrame(list(self.StorageData.loc["Investment cost (CHF/kWh)"]))
        if LinearCostStorage.empty is False:
            LinearCostStorage.columns=[1]
            linStorageCosts=self.Dict1D(linStorageCosts, LinearCostStorage)
        return linStorageCosts
    
    def StgOMInit(self):
        """
        Return Pyomo formatted list of O&M fixed cost per storage.
        """
        formatdata={}
        d=pd.DataFrame(list(self.StorageData.loc["Fixed O&M cost (CHF/kWh)"]))
        if d.empty is False:
            d.columns=[1]
            formatdata=self.Dict1D(formatdata, d)
        return formatdata
    
    def StgCO2Init(self):
        """
        Return Pyomo formatted list of CO2 fixed cost per storage.
        """
        formatdata={}
        d=pd.DataFrame(list(self.StorageData.loc["CO2 investment (kg-CO2/kWh)"]))
        if d.empty is False:
            d.columns=[1]
            formatdata=self.Dict1D(formatdata, d)
        return formatdata    

    
    def StgCRF(self):
        """
        Return Pyomo formatted CRF per storage.
        """
        CRFstg={}
        LifeBattery=pd.DataFrame(list(self.StorageData.loc["Lifetime (years)"]))
        if LifeBattery.empty is False:
            LifeBattery.columns=[1]
            CRF=1 / (((1 + self.interest) ** LifeBattery - 1) / (self.interest * ((1 + self.interest) ** LifeBattery)))
            CRFstg=self.Dict1D(CRFstg,CRF)
        return CRFstg
    
    def StgMaxMinCapInit(self):
        """
        Return Pyomo formatted dict of max and min cap per storage tech, hub and EC
        """

        Nstg = self.StorageData.shape[1] # number of storage techs
        dmax = {}
        dmin = {}
        maxcap = pd.DataFrame(list(self.StorageData.loc["Maximum capacity (kWh)"]))
        mincap = pd.DataFrame(list(self.StorageData.loc["Minimum capacity (kWh)"]))
        
        if maxcap.empty is False: # or mincap
        
            dmax = np.zeros((self.numberofhubs, self.numberec, Nstg)) # this format is required by DictPanel (not intuitive, could be improved)
            dmin = np.zeros((self.numberofhubs, self.numberec, Nstg)) 
                           
            for i in range(Nstg):
                for k in range(self.numberec):
                    for j in range(self.numberofhubs):
                        #check if technology is present in each hub
                        if isinstance(self.StorageData.loc["Hubs"][i],float) or isinstance(self.StorageData.loc["Hubs"][i],int):  # present in only one hub
                            if j+1==self.StorageData.loc["Hubs"][i] and k+1 == data.stgE[i]: #if hub j exists for tech i, AND the storage energy == energy carrier for tech i, then max cap is assigned; python starts with 0, Pyomo with 1
                                dmax[j,k,i] = maxcap.iloc[i][0]
                                dmin[j,k,i] = mincap.iloc[i][0]
                                
                        elif str(j+1) in list(self.StorageData.loc["Hubs"][i]) and k+1 == data.stgE[i]: #python starts with 0, Pyomo with 1; if hub j exists for tech i (in a list of multiple hubs), AND the storage energy == energy carrier for tech i, then max cap is assigned;
                            dmax[j,k,i] = maxcap.iloc[i][0]
                            dmin[j,k,i] = mincap.iloc[i][0]

        maxc = pd.Panel(dmax)
        maxc.major_axis = [int(x)+1 for x in maxc.major_axis]
        maxc.minor_axis = [int(x)+1 for x in maxc.minor_axis]
        maxcap_dict={}
        maxcap_dict = self.DictPanel(maxcap_dict, maxc)
        
        minc = pd.Panel(dmin)
        minc.major_axis = [int(x)+1 for x in minc.major_axis]
        minc.minor_axis = [int(x)+1 for x in minc.minor_axis]
        mincap_dict={}
        mincap_dict = self.DictPanel(mincap_dict, minc)
        
        return (maxcap_dict, mincap_dict)
    
    def StgCapInit(self, model):
        """
        Initialize existing storage capacities; return pyomo formatted dictionary for binary cost indicator
        """
        
        YN_stgcost_dict = {} # indicator to apply costs based on capacity (investment); model does not apply these costs to a pre-installed capacity
        setvals = [] # for tracking purposes only
        
        if self.instCapStg.empty is False:
            for i in range(len(self.instCapStg)): # for each tech
                stgID = i+1 # CHANGE electricity has ID == 1
                ecID = self.stgE[i]
                if(np.isnan(self.instCapStg[i])): # if capacity is not given
                    YN_stgcost_dict[stgID] = 1 # investment costs apply
                else:
                    YN_stgcost_dict[stgID] = 0 # capacity is pre-defined; investment costs do not apply
                
                    for j in range(self.numberofhubs): # for each hub
                        hubID = j+1
                        hublist = self.StorageData.loc["Hubs"][i]
                        if isinstance(hublist,float): # if only a single hub was given and it is read as float, convert to int
                            hublist = int(hublist)
                        hublist = str(hublist).split(',') # convert to string (int case); if more thane one hub is given, split hubs into a list using ',' as a delimiter
                        
                        if str(hubID) in hublist: # if capacity is given and technology is present in hub
                            capVal = self.instCapStg[i]
                            model.CapStg[hubID, stgID, ecID].fix(capVal)
                            setvals.append([hubID, stgID, ecID, capVal])
                            
        return (YN_stgcost_dict)

    #----------------------------------------------------------------------
    # Network 
    #----------------------------------------------------------------------
    
    def NetCRF(self):
        """
        Return Pyomo formatted NPV per storage.
        """
        Life=pd.DataFrame(list(self.NetworkData.loc["Lifetime (years)"]))
        CRFnet={}
        
        if Life.empty is False:
            Life.columns=[1]
            CRF=1 / (((1 + self.interest) ** Life - 1) / (self.interest * ((1 + self.interest) ** Life)))
            CRFnet=self.Dict1D(CRFnet,CRF)
            
        return CRFnet
    
     
    def Network_assign(self, model):
        """
        Assign network connection data per link, connection hubs, and energy carrier; return pyomo formatted dictionaries
        """
        
#        YN = np.zeros(data.NetworkData.shape[1], self.numberofhubs, self.numberofhubs, self.numberofdemands) # by default, no connection exists
        
        YNx_dict = {} # network link between hubs (indicator for allowable flow from hub i to j)
        len_dict = {} # network length
        loss_dict = {} # network loss
        invcost_dict = {} # investment cost
        OMFcost_dict = {} # fixed O&M cost
        OMVcost_dict = {} # variable O&M cost
        maxcap_dict = {} # maximum capacity
        mincap_dict = {} # minimum capacity
        CO2_dict = {} # CO2 factor
        life_dict = {} # lifetime
        YN_netcost_dict = {} # indicator to apply costs based on capacity (investment, fixed); model does not apply these costs to a pre-installed capacity, or from hub j to i where costs from hub i to j are already accounted for
                
        for i in range(data.NetworkData.shape[1]):
            linkID = i+1
            len_dict[linkID] = data.netLength.iloc[i]
            loss_dict[linkID] = data.netLoss.iloc[i]
            invcost_dict[linkID] = data.invCost.iloc[i]
            OMFcost_dict[linkID] = data.OMFCost.iloc[i]
            OMVcost_dict[linkID] = data.OMVCost.iloc[i]
            CO2_dict[linkID] = data.netCO2.iloc[i]
            life_dict[linkID] = data.netlife.iloc[i]
            for j in range(self.numberofhubs):
                hub_i = j+1
                for k in range(self.numberofhubs):
                    hub_j = k+1
                    for l in range(self.numberec):
                        EC = l+1
                        if data.node1.iloc[i] == hub_i and data.node2.iloc[i] == hub_j and data.netE[i] == EC and hub_i != hub_j:
                            YNx_dict[linkID, hub_i, hub_j, EC] = 1
                            
                            if data.uniFlow.iloc[i] == 'y':
                                YNx_dict[linkID, hub_j, hub_i, EC] = 0
                            else:
                                YNx_dict[linkID, hub_j, hub_i, EC] = 1
                                
                            maxcap_dict[linkID, hub_i, hub_j, EC] = data.maxCap.iloc[i]
                            maxcap_dict[linkID, hub_j, hub_i, EC] = data.maxCap.iloc[i]
                            mincap_dict[linkID, hub_i, hub_j, EC] = data.minCap.iloc[i]
                            mincap_dict[linkID, hub_j, hub_i, EC] = data.minCap.iloc[i]
                            
                            if data.instCap.iloc[i] > 0: # if installed capacity exists
                                model.CapNet[linkID, hub_i, hub_j, EC].fix(data.instCap.iloc[i]) # set capacity to data.instCap.iloc[i]
                                YN_netcost_dict[linkID, hub_i, hub_j, EC] = 0 # do not apply cost based on capacity (investment, fixed)
                                YN_netcost_dict[linkID, hub_j, hub_i, EC] = 0
                            else:
                                YN_netcost_dict[linkID, hub_i, hub_j, EC] = 1 # only acccount for investment and fixed cost from i to j
                                YN_netcost_dict[linkID, hub_j, hub_i, EC] = 0 # ignore investment and fixed cost from j to i
                            
                        elif data.node1.iloc[i] == hub_j and data.node2.iloc[i] == hub_i and data.netE[i] == EC and hub_i != hub_j:
                            linkID # do nothing (parameters have been set in previous block; this segment is to avoid overwriting values with 0)
                        
                        else:
                            YNx_dict[linkID, hub_i, hub_j, EC] = 0
                            maxcap_dict[linkID, hub_i, hub_j, EC] = 0
                            mincap_dict[linkID, hub_i, hub_j, EC] = 0
                            YN_netcost_dict[linkID, hub_i, hub_j, EC] = 0
                             # set capacity to 0
                            
        return (YNx_dict, YN_netcost_dict, len_dict, loss_dict, invcost_dict, OMFcost_dict, OMVcost_dict, maxcap_dict, mincap_dict, CO2_dict, life_dict)
    
#    def netLength_assign(self):
#        """
#        Return Pyomo formatted list of linear investment cost per storage.
#        """
#        linStorageCosts={}
#        LinearCostStorage=pd.DataFrame(list(self.StorageData.loc["CostBat (chf/kWh)"]))
#        LinearCostStorage.columns=[1]
#        linStorageCosts=self.Dict1D(linStorageCosts, LinearCostStorage)
#        return linStorageCosts        
        

# # Initialized class with excel input data

# In[28]:


excel_path=r'C:\Users\yam\Documents\Ehub modeling tool\python-ehub-dev\cases\Ehub input - multi-node example v5.3.xlsx'
data=InputData(excel_path)


# # Initialize optimization model

# ### Create model, sets, variables

# In[29]:


#-----------------------------------------------------------------------------#
## Creating a model ##
#-----------------------------------------------------------------------------#
fixednetwork=1
model = ConcreteModel()

### SETS ###

# sets definition
model.Time = RangeSet(1, data.numbertime) #0 is items, 1 is row
model.SubTime = RangeSet(2, data.numbertime, within=model.Time) #0 is items, 1 is row
#model.In = RangeSet(1, data.Technologies.shape[2]+1) # CHANGE replaced with model.Tech; 0 is items, 1 is row , 2 is column , it is assumed that the layout of hub's technologies is the same and the choice of technology is controled by max cap in each DF
#model.Out = RangeSet(1, data.numberofdemands) # CHANGE replaced with model.EC; 0 is items, 1 is row , 2 is column
model.Tech = RangeSet(1,data.numbertech) # set of technologies
model.EC = RangeSet(1, data.numberec) # set of energy carriers
#model.NonElectricity = Set(initialize=list(range(2,data.numberofdemands+1)), within=model.EC) # CHANGE remove
#number_of_demands= list(range(1, data.numberofdemands+1))

model.Stg = RangeSet(1, data.StorageData.shape[1])  # storage technology ID
#model.StgE = Set(initialize=set(data.stgE), within = model.Out) # unique set of energy carriers considered in storage

model.SolEC = Set(initialize=data.solECID, within=model.EC) # solar energy carrier ID

model.SolarTechs = Set(initialize=data.techSol, within=model.Tech)
#model.DispTechs = Set(initialize=data.DispTechsSet(), within=model.Tech) # CHANGE, remove, don't think this is needed anymore

#model.CHP = Set(initialize=data.CHP_list(), within=model.Tech) # set dispatch tech set
model.roof_tech = Set(initialize=data.Roof_tech(), within=model.Tech) # set tech with roof area set - MOVE TO CUSTOM CONSTRAINT MODULE

model.TechFixIn = Set(initialize=data.techFixIn, within=model.Tech) # set of technologies without a fixed output share
model.TechFixOut = Set(initialize=data.techFixOut, within=model.Tech) # set of technologies with a fixed output share
model.TechFreeOut = Set(initialize=data.techFreeOut, within=model.Tech) # set of technologies without a fixed output share
#model.TechFreeNoSol = Set(initialize=model.TechFreeOut-model.SolarTechs, within=model.Tech) # CHANGE remove once solar cap no longer in m2
#model.TechFreeSol = Set(initialize=model.TechFreeOut&model.SolarTechs, within=model.Tech) # CHANGE remove once solar cap no longer in m2
#model.TechFixNoSol = Set(initialize=model.TechFixOut-model.SolarTechs, within=model.Tech) # CHANGE remove once solar cap no longer in m2
#model.TechFixSol = Set(initialize=model.TechFixOut&model.SolarTechs, within=model.Tech) # CHANGE remove once solar cap no longer in m2
model.PartLFree = Set(initialize = model.TechFreeOut & data.techPL, within=model.Tech)
model.PartLFix = Set(initialize = model.TechFixOut & data.techPL, within=model.Tech)
model.PartLTech = Set(initialize = data.techPL, within=model.Tech)

model.EImpSet = Set(initialize=data.EImpInit(), within=model.EC) # set of import (supply) energy carriers
model.EExpSet = Set(initialize=data.EExpInit(), within=model.EC) # set of export energy carriers
model.ENImpSet = Set(initialize=set(model.EC) - set(model.EImpSet), within=model.EC) # set of energy carriers NOT imported
model.ENExpSet = Set(initialize=set(model.EC) - set(model.EExpSet), within=model.EC) # set of energy carriers NOT exported

#numberofhubs=data.numberofhubs
model.hubs=RangeSet(1,data.numberofhubs) #0 is items /number of hubs
model.hub_i=RangeSet(1,data.numberofhubs, within=model.hubs) #used for networks e.q. Q(i,j)
model.hub_j=RangeSet(1,data.numberofhubs, within=model.hubs) #used for networks e.q. Q(i,j)

model.LinkID = RangeSet(1,data.NetworkData.shape[1]) # link ID for network connection between hub_i and hub_j for the given energy carrier

### VARIABLES ###

## Technology variables
model.Ein = Var(model.hubs, model.Time, model.Tech, model.EC, domain=NonNegativeReals) #input energy/power (before efficiency)
model.Eout = Var(model.hubs, model.Time, model.Tech, model.EC, domain=NonNegativeReals) # output energy from a process/technology
model.Eexp = Var(model.hubs, model.Time, model.EC, domain=NonNegativeReals) #exported energy/power
model.Eimp= Var(model.hubs, model.Time, model.EC, domain=NonNegativeReals) #imported energy
model.CapTech = Var(model.hubs, model.Tech, domain=NonNegativeReals) #installed capacities per technologies
#model.Ytechnologies = Var(model.hubs, model.Tech, model.EC, domain=Binary) #binary if the technology has been installed - CHANGED: not needed
model.Ypl_op = Var(model.hubs, model.Time, model.PartLTech, domain=Binary) #binary for part-load showing if the technology is on or off 

#Storage variables
model.InStg = Var(model.hubs, model.Time, model.Stg, model.EC, domain=NonNegativeReals) #how much storage is charged
model.OutStg = Var(model.hubs, model.Time, model.Stg, model.EC, domain=NonNegativeReals) #how much storage is discharged
model.SoC = Var(model.hubs, model.Time, model.Stg, model.EC, domain=NonNegativeReals) #state of charge of storage
model.CapStg = Var(model.hubs, model.Stg, model.EC, domain=NonNegativeReals) #installed capacity of storage
#model.Ystorage = Var(model.hubs, model.Out, domain=Binary)
#model.maxStorageCap = Param(model.Out, initialize= maxStorageCap)

## Network variables
model.NetE = Var(model.LinkID, model.hub_i, model.hub_j, model.EC, model.Time, domain=NonNegativeReals)
model.Yx_op = Var(model.LinkID, model.hub_i, model.hub_j, model.EC, model.Time, domain=Binary)
model.CapNet = Var(model.LinkID, model.hub_i, model.hub_j, model.EC, domain=NonNegativeReals)

# Totals variables
model.TotalCost = Var(domain=Reals) #total cost
model.FuelCost = Var(domain=NonNegativeReals) # fuel cost
model.VOMCost = Var(domain=NonNegativeReals) # variable operation and maintainance cost
model.FOMCost = Var(domain=NonNegativeReals) # fixed operation and maintainance cost
model.CO2Tax = Var(domain=NonNegativeReals) # CO2 tax
model.IncomeExp = Var(domain=NonNegativeReals) #'earned' money from exporting 
model.InvCost = Var(domain=NonNegativeReals) #investment cost
model.TotalCarbon = Var(domain=Reals) #total carbon
model.TotalCarbon2 = Var(domain=Reals) #total carbon (due to Pyomo internal rules it is needed to have two variables)
model.FuelCO2 = Var(domain=Reals) # CO2 from ECs
model.TechCO2 = Var(domain=Reals) # CO2 from technology installation
model.StgCO2 = Var(domain=Reals) # CO2 from storage installation
model.NetCO2 = Var(domain=Reals) # CO2 from network installation

### PARAMETERS ###

## Technology parameters
model.eff = Param(model.TechFreeOut, initialize = data.EffInit()) # efficiencies for technologies without fixed output shares
model.effFixOut = Param(model.TechFixOut, model.EC, initialize = data.OutFixInit()) # efficiencies for technologies with fixed output shares
model.shrFixIn = Param(model.TechFixIn, model.EC, initialize = data.InFixInit())
model.shrFixOut = Param(model.TechFixIn, model.EC, initialize = data.techFixOut)
model.partLoad = Param(model.Tech, initialize=data.partload) #PartloadInput
model.lifeTechs = Param(model.Tech, initialize = data.LifeTime())
model.maxCapTechs = Param(model.hubs, model.Tech, initialize=data.TechLimitInit("Maximum capacity (kW)"))
model.minCapTechs = Param(model.hubs, model.Tech, initialize=data.TechLimitInit("Minimum capacity (kW)"))
model.maxOutTechs = Param(model.hubs, model.Tech, initialize=data.TechLimitInit("Maximum output (kWh)"))
model.minOutTechs = Param(model.hubs, model.Tech, initialize=data.TechLimitInit("Minimum output (kWh)"))
model.techCO2 = Param(model.Tech, initialize=data.techCO2) #PartloadInput
model.YtCapCost = Param(model.Tech, initialize = data.TechCapInit(model)) # Tech pre-installed capacities; indicator to apply costs based on capacity (investment); model does not apply these costs to a pre-installed capacity; this function also assigns pre-installed capacities to the cap variable
model.solkWm2 = Param(model.Tech, initialize = data.solkWm2)
model.invTech = Param(model.Tech, initialize= data.techInvCost) # Technologies capital costs
model.omvTech = Param(model.Tech, initialize=data.OMvar) # Variable operation and maintenance costs
model.omfTech = Param(model.Tech, initialize=data.OMfix) # Fixed operation and maintenance costs
model.CRFtech = Param(model.Tech, domain=NonNegativeReals, initialize=data.TechCRF()) # CRF for technologies

## Storage parameters
model.maxStorCh = Param(model.Stg, initialize=data.StorageCh()) # storage max charging rate
model.maxStorDisch = Param(model.Stg, initialize= data.StorageDisch()) # storage max discharging rate
model.standbyLoss = Param(model.Stg, initialize = data.StorageLoss()) # storage standby loss
model.chargingEff = Param(model.Stg, initialize = data.StorageEfCh()) # storage charging efficiency
model.dischargingEff = Param(model.Stg, initialize = data.StorageEfDisch()) # storage discharging efficiency
model.minSoC = Param(model.Stg, initialize = data.StorageMinSoC()) # storage min state of charge
model.invStg = Param(model.Stg, initialize = data.StorageLinCost()) # storage investement cost
model.lifeStg = Param(model.Stg, initialize = data.StorageLife()) # storage lifetime, not used
model.omfStg = Param(model.Stg, initialize = data.StgOMInit()) # storage O&M fixed cost
model.stgCO2 = Param(model.Stg, initialize = data.StgCO2Init()) # storage carbon factor for installation
model.CRFstg = Param(model.Stg, domain=NonNegativeReals, initialize=data.StgCRF()) # CRF for storage
maxS_dict, minS_dict = data.StgMaxMinCapInit()
model.maxCapStg = Param(model.hubs, model.Stg, model.EC, initialize = maxS_dict) # max installed storage capacity
model.minCapStg = Param(model.hubs, model.Stg, model.EC, initialize = minS_dict) # min installed storage capacity
model.YsCapCost = Param(model.Stg, initialize = data.StgCapInit(model)) # Storage pre-installed capacities initalize and flag

## Network parameters
YNx_dict, YNx_capcost_dict, len_dict, loss_dict, invcost_dict, OMFcost_dict, OMVcost_dict, maxcap_dict, mincap_dict, CO2_dict, life_dict =  data.Network_assign(model) # must be called after model.CapNet declaration (function initializes model.CapNet installed capacities)
model.netLength = Param(model.LinkID, initialize = len_dict) # network length
model.netLoss = Param(model.LinkID, initialize = loss_dict) # network loss
model.invNet = Param(model.LinkID, initialize = invcost_dict) # investment cost
model.omfNet = Param(model.LinkID, initialize = OMFcost_dict) # fixed O&M cost
model.omvNet = Param(model.LinkID, initialize = OMVcost_dict) # variable O&M cost
model.CRFnet = Param(model.LinkID, domain=NonNegativeReals, initialize=data.NetCRF()) # CRF for networks 
model.netMax = Param(model.LinkID, model.hub_i, model.hub_j, model.EC, initialize = maxcap_dict) # maximum capacity
model.netMin = Param(model.LinkID, model.hub_i, model.hub_j, model.EC, initialize = mincap_dict) # minimum capacity
model.netCO2 = Param(model.LinkID, initialize = CO2_dict) # network link installation CO2 factor
model.lifeNet = Param(model.LinkID, initialize = life_dict) # network link lifetime, not used
model.Yx = Param(model.LinkID, model.hub_i, model.hub_j, model.EC, initialize = YNx_dict) # network link between hubs (indicator for allowable flow from hub i to j)
model.YxCapCost = Param(model.LinkID, model.hub_i, model.hub_j, model.EC, initialize = YNx_capcost_dict) # indicator to apply costs based on capacity (investment, fixed); model does not apply these costs to a pre-installed capacity, or from hub j to i where costs from hub i to j are already accounted for

## Import/export parameters
model.impCost = Param(model.EImpSet, initialize=data.ImpLookup("Price (CHF/kWh)"))    # fuel price
model.expPrice = Param(model.EExpSet, initialize=data.ExpPriceInit())  # Export price
model.co2Tax = Param(model.EImpSet, initialize=data.ImpLookup("CO2 tax (CHF/kg-CO2)"))    # carbon tax 
model.maxImp = Param(model.EImpSet, initialize=data.ImpLookup("Maximum supply (kWh)"))    # max import supply
model.ecCO2 = Param(model.EImpSet, initialize=data.ImpLookup("CO2 (kg-CO2/kWh)"))

## Other global parameters
model.interestRate = Param(within=NonNegativeReals, initialize=data.interest) # not used
model.bigM = Param(within=NonNegativeReals, initialize=data.bigm) # acts as an upper limit for Eout from part load technologies and network link exchanges
model.maxCarbon = Param(initialize=data.maxcarbon)
model.loads = Param(model.hubs, model.Time, model.EC, initialize=data.Demands())
model.solarEm = Param( model.Time, initialize=data.SolarData())
model.maxSolarArea = Param(initialize=500) # - MOVE TO CUSTOM CONSTRAINT MODULE

# In[30]:

#-----------------------------------------------------------------------------#
# Constraint defintion
#-----------------------------------------------------------------------------#

#-----------------------------------------------------------------------------#
# Energy balance
#-----------------------------------------------------------------------------#
def loadsBalance_rule(model, h,  m, e): #energy balance when multiple hubs are present
    return (model.loads[h,m,e] + model.Eexp[h,m,e] == (model.Eimp[h, m, e] + sum(model.OutStg[h,m,s,e] - model.InStg[h,m,s,e] for s in model.Stg) +
                                                        sum(model.Eout[h,m,t,e] - model.Ein[h,m,t,e] for t in model.Tech) 
                                                                + sum(model.NetE[l,j,h,e,m]*(1-model.netLoss[l]*model.netLength[l])- model.NetE[l,h,j,e,m] for l in model.LinkID for j in model.hubs ))) 

model.loadsBalance = Constraint(model.hubs, model.Time, model.EC, rule=loadsBalance_rule)

#-----------------------------------------------------------------------------#
# Technology energy and capacity constraints
#-----------------------------------------------------------------------------#

# Output energy (Eout) for technologies without a fixed output share (model.TechFreeOut)
def EoutFree_rule(model, h, m, t):
    return (sum(model.Eout[h,m,t,e] for e in model.EC) == sum(model.Ein[h,m,t,e] for e in model.EC)*model.eff[t])
model.eoutFree = Constraint(model.hubs, model.Time, model.TechFreeOut, rule=EoutFree_rule)

# Output energy (Eout) for technologies with a fixed output share (model.TechFixOut)
def EoutFix_rule(model, h, m, t, e):
    return (model.Eout[h,m,t,e] == sum(model.Ein[h,m,t,ec] for ec in model.EC)*model.effFixOut[t,e])
model.eoutFix = Constraint(model.hubs, model.Time, model.TechFixOut, model.EC, rule=EoutFix_rule)

# Output energy (Eout) for technologies with a fixed output share (model.TechFixOut)
def EinFix_rule(model, h, m, t, e):
    return (model.Ein[h,m,t,e] == sum(model.Ein[h,m,t,ec] for ec in model.EC)*model.shrFixIn[t,e])
model.einFix = Constraint(model.hubs, model.Time, model.TechFixIn, model.EC, rule=EinFix_rule)   

# set Eout and Ein == 0 for ECs that do not exist for each technology
#stin = [] # for tracking only
for t in range(1,data.numbertech+1):
    necin = set(model.EC) - set(data.inEC[t])
    if t not in model.TechFixIn:
        for e in necin:
            for h in range(1,data.numberofhubs+1):
                for m in range(1,data.numbertime+1):
                    model.Ein[h,m,t,e].fix(0)
#                    stin.append([h,m,t,e])
    
    if t not in model.TechFixOut:
        necout = set(model.EC) - set(data.outEC[t])
        for e in necout:
            for h in range(1,data.numberofhubs+1):
                for m in range(1,data.numbertime+1):
                     model.Eout[h,m,t,e].fix(0)

#technology output cannot be higher than installed capacity 
# - for technologies without fixed output shares:
def capConstFree_rule(model, h, m, t):
    return (sum(model.Eout[h,m,t,e] for e in model.EC) <= model.CapTech[h,t])
model.capConstFree = Constraint(model.hubs, model.Time, model.TechFreeOut, rule=capConstFree_rule)

# - for technologies with fixed output shares:
model.capConstFix= ConstraintList()
for t in set(model.TechFixOut):
    for h in set(model.hubs):
        for m in set(model.Time):
            model.capConstFix.add(model.Eout[h,m,t,data.outEC[t][0]] <= model.CapTech[h,t]) # constraint applies to first output EC only (remaining output ECs determined according to given fixed shares)

# installed capacity must be less than maximum capacity (max cap defaults to infinity if not specified)
def maxCapacity_rule(model, h, t):
    return (model.CapTech[h,t] <= model.maxCapTechs[h,t])
model.maxCapacity = Constraint(model.hubs, model.Tech, rule=maxCapacity_rule)

# installed capacity must be greater than minimum capacity (min cap defaults to zero if not specified)
def minCapacity_rule(model, h, t):
    return (model.CapTech[h,t] >= model.minCapTechs[h,t])
model.minCapacity = Constraint(model.hubs, model.Tech, rule=minCapacity_rule)

# total technology output cannot be higher than maximum output
# - for technologies without fixed output shares:
def maxOutConstFree_rule(model, h, t):
    return (sum(model.Eout[h,m,t,e] for m in model.Time for e in model.EC) <= model.maxOutTechs[h,t])
model.maxOutConstFree = Constraint(model.hubs, model.TechFreeOut, rule=maxOutConstFree_rule)

# - for technologies with fixed output shares:
model.maxOutConstFix= ConstraintList()
for t in set(model.TechFixOut):
    for h in set(model.hubs):
        model.maxOutConstFix.add(sum(model.Eout[h,m,t,data.outEC[t][0]] for m in model.Time) <= model.maxOutTechs[h,t]) # constraint applies to first output EC only (remaining output ECs determined according to given fixed shares)

# total technology output cannot be less than minimum output
# - for technologies without fixed output shares:
def minOutConstFree_rule(model, h, t):
    return (sum(model.Eout[h,m,t,ec] for m in model.Time for ec in model.EC) >= model.minOutTechs[h,t])
model.minOutConstFree = Constraint(model.hubs, model.TechFreeOut, rule=minOutConstFree_rule)

# - for technologies with fixed output shares:
model.minOutConstFix= ConstraintList()
for t in set(model.TechFixOut):
    for h in set(model.hubs):
        model.minOutConstFix.add(sum(model.Eout[h,m,t,data.outEC[t][0]] for m in model.Time) >= model.minOutTechs[h,t]) # constraint applies to first output EC only (remaining output ECs determined according to given fixed shares)

#-----------------------------------------------------------------------------#
# Import/export constraints
#-----------------------------------------------------------------------------#

# maximum import supply
def maxEimp_rule(model, e):
    return(sum(model.Eimp[h,m,e] for h in model.hubs for m in model.Time) <= model.maxImp[e])
model.maxEimpConst = Constraint(model.EImpSet, rule=maxEimp_rule)

# set non-export and non-import ECs to zero
def Eexp_rule(model, h, m, e):
    return (model.Eexp[h,m,e] == 0)
model.ExpConst = Constraint(model.hubs, model.Time, model.ENExpSet, rule=Eexp_rule) # set non-export ECs to zero

def Eimp_rule(model, h, m, e):
    return (model.Eimp[h,m,e] == 0)
model.ImpConst = Constraint(model.hubs, model.Time, model.ENImpSet, rule=Eimp_rule) # set non-import ECs to zero

#-----------------------------------------------------------------------------#
# Carbon constraints
#-----------------------------------------------------------------------------#

def carbonConst_rule(model):
    return (model.TotalCarbon <= model.maxCarbon)

if np.isnan(data.maxcarbon) == False: # if max carbon is not nan (i.e., is given)
    model.carbonConst = Constraint(rule=carbonConst_rule) #use for epsilon contraint multi objective


# ### Create technology specific constraints

# In[31]:


#-----------------------------------------------------------------------------#
## Specific constraints 
#-----------------------------------------------------------------------------#


# lower bound for part load
# - for technologies without fixed output shares:
def partLoadFreeL_rule(model, h, m, plt):
    return (model.partLoad[plt] * model.CapTech[h, plt] <= sum(model.Eout[h,m,plt,e] for e in model.EC) + model.bigM * (1 - model.Ypl_op[h, m, plt]))

model.partLoadFreeL = Constraint(model.hubs, model.Time, model.PartLFree, rule=partLoadFreeL_rule)

# - for technologies with fixed output shares:
model.partLoadFixL= ConstraintList()
for plt in set(model.PartLFix):
    for h in set(model.hubs):
        for m in set(model.Time):
            model.partLoadFixL.add(model.partLoad[plt] * model.CapTech[h, plt] <= model.Eout[h,m,plt,data.outEC[plt][0]] + model.bigM * (1 - model.Ypl_op[h, m, plt])) # constraint applies to first output EC only (remaining output ECs determined according to given fixed shares)

#upper bound for part load
def partLoadU_rule(model, h, m, plt):    
    return (sum(model.Eout[h,m,plt,e] for e in model.EC) <= model.bigM * model.Ypl_op[h,m,plt])

model.partLoadU = Constraint(model.hubs, model.Time, model.PartLTech, rule=partLoadU_rule)

#solar output is equal to the installed capacity
def solarInput_rule(model, h, m, st, se):
    return (model.Ein[h,m,st,se] == model.solarEm[m] * model.CapTech[h,st] / model.solkWm2[st])

model.solarInput = Constraint(model.hubs, model.Time, model.SolarTechs, model.SolEC, rule=solarInput_rule) 

#sum of roof area of all roof techs cannot be bigger than total roof area - MOVE TO CUSTOM CONSTRAINT MODULE, user should manually indicate which techs are roof top (e.g., PV panel could be for ground)
def roofArea_rule(model,h):
    return (sum(model.CapTech[h,rt] / model.solkWm2[rt] for rt in model.roof_tech) <= model.maxSolarArea) # Roof area of all roof technologies has to be smaller than the total roof area
model.roofArea = Constraint(model.hubs,rule=roofArea_rule) #model.roof_tech

# REMOVED Ytechnologies
#if tech is installed, Ytechnologies binary is 1 (it can be used for fixed investment costs)
#def fixCostConst_rule1(model, i, inp, out):
#    return (model.CapTech[i, inp,out] <= model.bigM * model.Ytechnologies[i, inp,out])
#model.fixCostConst1 = Constraint(model.hubs, model.Tech, model.EC, rule=fixCostConst_rule1)
#
#def fixCostConst_rule2(model, i, inp, out):
#    return (model.Ytechnologies[i, inp,out] <= model.CapTech[i, inp,out]) # actually, this enforces that if tech is installed, cap must be >= 1 kW... not a good solution
#model.fixCostConst2 = Constraint(model.hubs, model.Tech, model.EC, rule=fixCostConst_rule2)

#define which two outputs a CHP is producing, the installed capacity is the first output (electricity usually)
#dispatch_demands=data.DisDemands()  
#CHP_list=data.CHP_list()
#model.con = ConstraintList() #because of Pyomo when assigning which ouputs CHP produces, it has to be manually assigned per CHP per output
#for i in range(1, numberofhubs+1):
#    for x in range(1, len(dispatch_demands)+1):
#        model.con.add(model.CapTech[i, CHP_list[x-1],dispatch_demands[x-1,1]] == model.cMatrix[CHP_list[x-1],dispatch_demands[x-1,1]] / model.cMatrix[CHP_list[x-1],dispatch_demands[x-1,0]] * model.CapTech[i, CHP_list[x-1],dispatch_demands[x-1,0]])
#        model.con.add(model.Ytechnologies[i, CHP_list[x-1],dispatch_demands[x-1,0]]==model.Ytechnologies[i, CHP_list[x-1],dispatch_demands[x-1,1]])
#        model.con.add(model.CapTech[i, CHP_list[x-1],dispatch_demands[x-1,0]] <= model.maxCapTechs[i, CHP_list[x-1]] * model.Ytechnologies[i, CHP_list[x-1],dispatch_demands[x-1,0]])

# In[31]:
        
#-----------------------------------------------------------------------------#
## Network constraints
#-----------------------------------------------------------------------------#

# network energy transfer must be less than energy output from capacity per hour
def netQ_rule(model, l, hi, hj, e, m):
    return (model.NetE[l,hi,hj,e,m] <= model.Yx[l,hi,hj,e] * model.CapNet[l,hi,hj,e])

model.netQ_const = Constraint(model.LinkID, model.hub_i, model.hub_j, model.EC, model.Time, rule=netQ_rule)

# sets Yx_op = 1 if flow exists
def YxOp_rule(model, l, hi, hj, e, m):
    return (model.NetE[l,hi,hj,e,m] <= (model.Yx_op[l,hi,hj,e,m] * model.bigM) )

model.YxOprule1_const = Constraint(model.LinkID, model.hub_i, model.hub_j, model.EC, model.Time, rule=YxOp_rule)

# sets Yx_op = 0 if flow = 0
#def YxOp_rule2(model, l, hi, hj, e, m):
#    return (model.Yx_op[l,hi,hj,e,m] <= model.NetE[l,hi,hj,e,m] * 10000) # FIX! ideally, thought it should be DH_Q*bigM to account for if DHQ is a fraction < 1, but doing this gives me an infeasibility; why? something is up here
#
#model.YxOprule2_const = Constraint(model.LinkID, model.hub_i, model.hub_j, model.EC, model.Time, rule=YxOp_rule2)

# ensure single directional flow at every time period (either flow from i to j or j to i, but not both in the same time period)
def netflow_rule(model, l, hi, hj, e, m):
    return (model.Yx_op[l,hi,hj,e,m] + model.Yx_op[l,hj,hi,e,m] <= 1)

model.netflow_const = Constraint(model.LinkID, model.hub_i, model.hub_j, model.EC, model.Time, rule=netflow_rule)

# assumed bi-directional network flow; capacity from i to j is equal to capacity from j to i
def netCapEq_rule(model, l, hi, hj, e):
    return (model.CapNet[l,hi,hj,e] == model.CapNet[l,hj,hi,e])

model.netCapEq_const = Constraint(model.LinkID, model.hub_i, model.hub_j, model.EC, rule=netCapEq_rule)

# ensure network minimum and maximum capacity bounds are obeyed
def netMaxCap_rule(model, l, hi, hj, e):
    return (model.CapNet[l,hi,hj,e] <= model.netMax[l,hi,hj,e])

model.netMaxCap_const = Constraint(model.LinkID, model.hub_i, model.hub_j, model.EC, rule=netMaxCap_rule)

def netMinCap_rule(model, l, hi, hj, e):
    return (model.CapNet[l,hi,hj,e] >= model.netMin[l,hi,hj,e])

model.netMinCap_const = Constraint(model.LinkID, model.hub_i, model.hub_j, model.EC, rule=netMinCap_rule)


# In[32]:

# ### Create storage constraints
#-----------------------------------------------------------------------------#
## Storage constraints
#-----------------------------------------------------------------------------#

#continuinity equation for storage
def storageBalance_rule(model, h, m, stg, e):
    return (model.SoC[h,m,stg,e] == ((1-model.standbyLoss[stg]) * model.SoC[h,(m-1),stg,e]  
                                + model.chargingEff[stg] * model.InStg[h,m,stg,e] 
                                - (1/model.dischargingEff[stg]) * model.OutStg[h,m,stg,e]))
model.storageBalance = Constraint(model.hubs, model.SubTime, model.Stg, model.EC, rule=storageBalance_rule) #storage continuinity variable

# storage state of charge (SoC) should be the same at the start and end time period
def storageStartEnd_rule(model, h, stg, e):
    return (model.SoC[h,1,stg,e] == model.SoC[h,data.numbertime,stg,e])
model.storageStartEnd = Constraint(model.hubs, model.Stg, model.EC, rule=storageStartEnd_rule)

# storage initial SoC
def initSoc_rule(model, h, stg, e):
    return (model.SoC[h,1,stg,e] == model.CapStg[h,stg,e]*model.minSoC[stg])
model.initSoC = Constraint(model.hubs, model.Stg, model.EC, rule=initSoc_rule)

#uncomment for different storage initializations
'''
model.StorCon = ConstraintList() #different storage initializations (arbitrarily)
for x in range(1, DemandData.shape[1]+1):#for applying to all storages
    
    #model.StorCon.add(model.SoC[1, x] == model.CapStg[x] * model.minSoC[x])
    model.StorCon.add(model.SoC[1, x] == model.SoC[8760, x])
    #model.StorCon.add(model.OutStg[1, x] == 0)
    


def storageInitBattery_rule(model):  #different storage initializations (arbitrarily)
    return (model.SoC[1, 1] == model.CapStg[1] * model.minSoC[1])
model.storageInitBattery = Constraint(rule=storageInitBattery_rule)

def storageInitThermal1_rule(model): #different storage initializations (arbitrarily)
    return (model.SoC[1, 2] == model.SoC[8760, 2])
model.storageInitThermal1 = Constraint(rule=storageInitThermal1_rule)

def storageInitThermal2_rule(model): #different storage initializations (arbitrarily)
    return (model.OutStg[1, 2] == 0)
model.storageInitThermal2 = Constraint(rule=storageInitThermal2_rule)
'''


def storageChargeRate_rule(model, h, m, stg, e):
    return (model.InStg[h,m,stg,e] <= model.maxStorCh[stg] * model.CapStg[h,stg,e])
model.storageChargeRate = Constraint(model.hubs, model.Time, model.Stg, model.EC, rule=storageChargeRate_rule) #maximum charging

def storageDischRate_rule(model, h, m, stg, e):
    return (model.OutStg[h,m,stg,e] <= model.maxStorDisch[stg] * model.CapStg[h,stg,e])
model.storageDischRate = Constraint(model.hubs,model.Time, model.Stg, model.EC, rule=storageDischRate_rule) #maximum discharging

# storage minimum SoC
def storageMinState_rule(model, h, m, stg, e):
    return (model.SoC[h,m,stg,e] >= model.CapStg[h,stg,e]*model.minSoC[stg])
model.storageMinState = Constraint(model.hubs, model.Time, model.Stg, model.EC, rule=storageMinState_rule) #minimum SoC allowed

#SoC has to be <=than installed capacity
def storageCap_rule(model, h, m, stg, e):
    return (model.SoC[h,m,stg,e] <= model.CapStg[h,stg,e])
model.storageCap = Constraint(model.hubs,model.Time, model.Stg, model.EC, rule=storageCap_rule) #smaller than storage cap

# maximum storage capacity constraint
def storageMaxCap_rule(model, h, stg, e):
    return (model.CapStg[h,stg,e] <= model.maxCapStg[h,stg,e])
model.storageMaxCap = Constraint(model.hubs, model.Stg, model.EC, rule=storageMaxCap_rule)

# minimum storage capacity constraint
def storageMinCap_rule(model, h, stg, e):
    return (model.CapStg[h,stg,e] >= model.minCapStg[h,stg,e])
model.storageMinCap = Constraint(model.hubs, model.Stg, model.EC, rule=storageMinCap_rule)

# ### Create objective function(s)

# In[33]:


#-----------------------------------------------------------------------------#
## Objective functions
#-----------------------------------------------------------------------------#

def objcost_rule(model):
    return (model.TotalCost)

def objcarbon_rule(model):
    return (model.TotalCarbon2)

if data.minobj == "Cost":
    model.Total_Cost = Objective(rule=objcost_rule, sense=minimize) # cost minimization objective
elif data.minobj == "Carbon":
    model.Total_Carbon = Objective(rule=objcarbon_rule, sense=minimize) # carbon minimization objective

#operational costs
def fuelCost_rule(model):
    return(model.FuelCost == (sum(model.impCost[ei] * model.Eimp[h, m, ei] for h in model.hubs for m in model.Time for ei in model.EImpSet)))
model.fuelCost = Constraint(rule=fuelCost_rule) 

def vomCost_rule(model):
    return(model.VOMCost == (sum(model.Eout[h,m,t,e] * model.omvTech[t]
                            for h in model.hubs for m in model.Time for t in model.Tech for e in model.EC)
                            + sum(model.NetE[l,hi,hj,e,m]*(1-model.netLoss[l]*model.netLength[l])*model.omvNet[l]
                            for l in model.LinkID for hi in model.hub_i for hj in model.hub_j for e in model.EC for m in model.Time)))
model.vomCost = Constraint(rule=vomCost_rule)

def fomCost_rule(model):
    return(model.FOMCost == (sum(model.CapTech[h,t]*model.omfTech[t] for h in model.hubs for t in model.Tech)
                            + sum(model.CapNet[l,hi,hj,e]*model.omfNet[l]
                            for l in model.LinkID for hi in model.hub_i for hj in model.hub_j for e in model.EC)
                            + sum(model.CapStg[h,s,e]*model.omfStg[s] for h in model.hubs for s in model.Stg for e in model.EC)))
model.fomCost = Constraint(rule=fomCost_rule)

def co2Tax_rule(model):
    return(model.CO2Tax == (sum(model.co2Tax[ei] * model.ecCO2[ei] * model.Eimp[h, m, ei] for h in model.hubs for m in model.Time for ei in model.EImpSet)))
model.co2TaxConst = Constraint(rule=co2Tax_rule)

#revenue from exporting
def incomeExp_rule(model):
    return(model.IncomeExp == (sum(model.expPrice[ex] * 
                            sum(model.Eexp[h,m,ex] for h in model.hubs for m in model.Time) 
                            for ex in model.EExpSet)))
model.incomeExp = Constraint(rule=incomeExp_rule)

#investment cost
def invCost_rule(model):
    return(model.InvCost == (sum(model.CRFtech[t] * (model.invTech[t] * model.CapTech[h,t] * model.YtCapCost[t]) for h in model.hubs for t in model.Tech)
                            + sum(model.CRFstg[stg] * model.invStg[stg] * model.CapStg[h,stg,e] * model.YsCapCost[stg] for h in model.hubs for stg in model.Stg for e in model.EC)
                            + sum(model.CRFnet[l]*model.invNet[l]*model.netLength[l]*model.CapNet[l,hi,hj,e]*model.YxCapCost[l,hi,hj,e] for l in model.LinkID for hi in model.hub_i for hj in model.hub_j for e in model.EC)))

model.invCost = Constraint(rule=invCost_rule)

#total cost
def totalCost_rule(model):
    return(model.TotalCost == model.InvCost + model.FuelCost + model.VOMCost + model.FOMCost + model.CO2Tax - model.IncomeExp)
model.totalCost = Constraint(rule=totalCost_rule) 

# CO2 from energy carriers
def fuelCO2_rule(model):
    return(model.FuelCO2 == sum(model.ecCO2[ei] * model.Eimp[h,m,ei] for h in model.hubs for m in model.Time for ei in model.EImpSet))
model.fuelCO2Const = Constraint(rule=fuelCO2_rule)

# CO2 from tech installation
def techCO2_rule(model):
    return(model.TechCO2 == sum(model.techCO2[t] * model.CapTech[h,t] * model.YtCapCost[t] for h in model.hubs for t in model.Tech))
model.techCO2Const = Constraint(rule=techCO2_rule)

# CO2 from storage installation
def stgCO2_rule(model):
    return(model.StgCO2 == sum(model.stgCO2[s] * model.CapStg[h,s,e] * model.YsCapCost[s] for h in model.hubs for s in model.Stg for e in model.EC))
model.stgCO2Const = Constraint(rule=stgCO2_rule)

# CO2 from network installation
def netCO2_rule(model):
    return(model.NetCO2 == sum(model.netCO2[l] * model.CapNet[l,hi,hj,e] *model.netLength[l] * model.YxCapCost[l,hi,hj,e] for l in model.LinkID for hi in model.hub_i for hj in model.hub_j for e in model.EC))
model.netCO2Const = Constraint(rule=netCO2_rule)

#Pyomo specific way to specify total carbon variable
def totalCarbon2_rule(model):
    return(model.TotalCarbon2 == model.TotalCarbon)
model.totalCarbon2 = Constraint(rule=totalCarbon2_rule)

#Pyomo specific way to specify total carbon variable
def totalCarbon_rule(model):
    return(model.TotalCarbon == model.FuelCO2 + model.TechCO2 + model.StgCO2 + model.NetCO2)
model.totalCarbon = Constraint(rule=totalCarbon_rule)

# # Solve optimisation model

# In[54]:


#-----------------------------------------------------------------------------#
## Solve model ##
#-----------------------------------------------------------------------------#


opt = SolverFactory("gurobi") #select solver
#opt.options["mipgap"]=0.05 #different options to use for solver (parameter name can be different depending on solver)
opt.options["FeasibilityTol"]=1e-05
opt.options['outlev']  = 1  # tell gurobi to be verbose with output
opt.options['iisfind'] = 1  # tell gurobi to find an iis table for the infeasible model
solver_manager = SolverManagerFactory("serial")
#results = solver_manager.solve(instance, opt=opt, tee=True,timelimit=None, mipgap=0.1) #this is gurobi syntax

model.preprocess()

results = solver_manager.solve(model, opt=opt, tee=True,timelimit=None) #this is gurobi syntax


#Example of how to print variables
#print(instance.TotalCost.value)
#print(model.TotalCost.value)



# # Usefull commands/syntax for debuging model/accessing Pyomo values

# In[35]:


#print(model.TotalCarbon.value)

with open('C:/Users/yam/Documents/GitHub/python-ehub/cases/Results/Results.txt', 'w') as f:
    f.write ('{} {}\n'.format("objective ", value(model.TotalCost)))
    for v in model.component_objects(Var, active=True):
        varobject = getattr(model, str(v))
        for index in varobject:
            if index is not None:
                line = ' '.join(str(t) for t in index)
                f.write ('{} {} {}\n'.format(v, line, varobject[index].value))
            else:
                f.write ('{} {} {}\n'.format(v, index, varobject[index].value))


# In[36]:


#! bokeh serve --show vis_class.py --port 50040


# In[37]:


#model.Eexp.get_values() #display variable values


# In[38]:


#model.NetE[5, 7, 1,4].value #display variable value for specific time/node/etc


# In[39]:


#model.loads[7, 1,4] # display parameter value for specific time/node/etc


# In[40]:


#sum(model.P[8, 1,inp].value*model.cMatrix[inp,3] for inp in model.In) #get variable value for specific time/node/etc


# In[41]:


#model.export.display() #display constraint values


# In[42]:


#model.cMatrix.display() #display constraint values


# # Access results and plot

# In[43]:


#input (fuel) for each technology
#P_matrix=np.zeros(shape=(DemandData.shape[0],Technologies.shape[1]+1))
#for i in range(1,DemandData.shape[0]+1):
#    for j in range(1,Technologies.shape[1]+1+1):
#        P_matrix[i-1,j-1]=model.P[i,j].value


# In[44]:


#access energy transfer for each pipe

#DH_matrix = [[0] * data.numberofhubs for i in range(data.DemandData.shape[1])]
#for i in range(data.numberofhubs):
#    for j in range(i+1, data.numberofhubs):
#        dummy =[]
#        for k in range(data.DemandData.shape[1]):
#            dummy.append(model.NetE[i+1,j+1,k+1,4].value) #change '4' to number of demands
                           
           
#        DH_matrix[i][j] = dummy
        
            
#DH = pd.DataFrame(DH_matrix)  
#DH = DH.replace(to_replace='NaN', value=0)
#DH.columns = range(1,len(DH.columns)+1)


# In[45]:


#energy balance for each node/hub
#node_matrix=[]
#dummy_timestep=[0]*8760 #change 8760 to number of timesteps considered
#for i in range(18):
#    node_matrix.append(dummy_timestep)
#    for j in range(18): #change 18 to number of nodes
#        if DH_matrix[i][j]<> 0:
#            node_matrix[i]=map(add,node_matrix[i],DH_matrix[i][j])
#        if DH_matrix[j][i]<>0:
#            node_matrix[i]=map(add,node_matrix[i],DH_matrix[j][i])


# In[46]:


#plt.plot(node_matrix[3]) #plot


# In[47]:


#sio.savemat('nodes', {'nodes':node_matrix}) #save results as Matlab .mat


# In[48]:


#df=pd.DataFrame(node_matrix)
#df.columns =list(range(1,8761))
#df.to_excel('nodes.xlsx',header=True,index=True) #save as excel


# In[49]:


##save energy profile for each pipe separately as xcel
#for i in range(18): #change 18 to number of nodes
#    for j in range(18): #change 18 to number of nodes
#        if DH_matrix[i][j]<>0:
#            if sum(DH_matrix[i][j])<>0:
#                df=pd.DataFrame(DH_matrix[i][j])
#                df.index = range(1,8761) #change 8761 to number of timesteps +1
#                df.columns=[str(i+1)+'-'+str(j+1)]
#                df.to_excel('pipe'+str(i+1)+'_'+str(j+1)+'.xlsx',header=True,index=True) #save energy profile for each pipe separately as excel


# In[50]:


#plt.plot(DH_matrix[2][4])


# In[51]:


#get installed capacities in all hubs/nodes for all demands
#capacities_matrix = np.zeros(((data.Technologies.shape[2]+1)*data.numberec, data.numberofhubs))
#for i in range(data.numberofhubs):
#    for j in range(data.Technologies.shape[2]):
#        capacities_matrix[j*5,i] = model.CapTech[i+1,j+1].value   #change 4 to number of demands
#            
#Capacities = pd.DataFrame(capacities_matrix)  
#Capacities = Capacities.replace(to_replace='NaN', value=0)
#Capacities.columns = range(1,len(Capacities.columns)+1)
#Capacities
#
#       
##get installed storage capacities in all hubs/nodes for all demands
#stgcapacities_matrix = np.zeros(((data.StorageData.shape[1])*data.numberec, data.numberofhubs))
#for i in range(data.numberofhubs):
#    for j in range(data.StorageData.shape[1]):
#        for k in range(data.numberec):
#            stgcapacities_matrix[j*3+k,i] = model.CapStg[i+1,j+1,k+1].value   #change 4 to number of demands
#            
#StgCapacities = pd.DataFrame(stgcapacities_matrix)  
#StgCapacities = StgCapacities.replace(to_replace='NaN', value=0)
#StgCapacities.columns = range(1,len(StgCapacities.columns)+1)
#StgCapacities


# In[52]:


#operation for all timesteps for all nodes/hubs

#operation_dict={}
#for i in range(data.numberofhubs):
#    matrix=np.zeros((data.DemandData.shape[1],data.Technologies.shape[2]+1))
#    for j in range(data.DemandData.shape[1]):
#        for k in range(data.Technologies.shape[2]+1):
#            matrix[j,k] = model.P[i+1,j+1,k+1].value
#    operation_dict[i]=matrix
#operation = pd.Panel(operation_dict)

