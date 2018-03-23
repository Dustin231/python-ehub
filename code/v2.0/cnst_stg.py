#-----------------------------------------------------------------------------#
# Storage Constraints
#-----------------------------------------------------------------------------#

from pyomo.core import *

def const_stg(data, model):
    
    print("- Storage constraints")
    
    # storage energy balance (state-of-charge)
    def storageBalance_rule(model, h, m, stg, e):
        return (model.SoC[h,m,stg,e] == ((1-model.standbyLoss[stg]) * model.SoC[h,(m-1),stg,e]  
                                    + model.chargingEff[stg] * model.InStg[h,m,stg,e] 
                                    - (1/model.dischargingEff[stg]) * model.OutStg[h,m,stg,e]))
    model.storageBalance = Constraint(model.hubs, model.Time, model.Stg, model.EC, rule=storageBalance_rule) #storage continuinity variable
    
    # storage state of charge (SoC) should be the same at the start and end time period
    def storageStartEnd_rule(model, h, stg, e):
        return (model.SoC[h,0,stg,e] == model.SoC[h,data.nTime,stg,e])
    model.storageStartEnd = Constraint(model.hubs, model.Stg, model.EC, rule=storageStartEnd_rule)
    
    # storage initial SoC
    def initSoc_rule(model, h, stg, e):
        return (model.SoC[h,0,stg,e] == model.CapStg[h,stg,e]*model.minSoC[stg]) # SoC at time 0
    model.initSoC = Constraint(model.hubs, model.Stg, model.EC, rule=initSoc_rule)
    
    # set operational binary indicatory for storage in/out flow
    def YstgIn_rule(model, h, m, stg, e):
        return (model.InStg[h,m,stg,e] <= (model.YstgIn[h,m,stg,e] * model.bigM))
    model.YstgIn_const = Constraint(model.hubs, model.Time, model.Stg, model.EC, rule=YstgIn_rule)
    
    def YstgOut_rule(model, h, m, stg, e):
        return (model.OutStg[h,m,stg,e] <= (model.YstgOut[h,m,stg,e] * model.bigM))
    model.YstgOut_const = Constraint(model.hubs, model.Time, model.Stg, model.EC, rule=YstgOut_rule)
    
    # ensure single directional flow at every time period (either storage in or out, but not both in the same time period)
    def stgflow_rule(model, h, m, stg, e):
        return (model.YstgIn[h,m,stg,e] + model.YstgOut[h,m,stg,e] <= 1)
    model.stgflow_const = Constraint(model.hubs, model.Time, model.Stg, model.EC, rule=stgflow_rule)
    
#    # storage InStg and OutStg == 0 in first time step
#    def initInStg_rule(model, h, stg, e):
#        return (model.InStg[h,1,stg,e] == 0)
#    model.initInStg = Constraint(model.hubs, model.Stg, model.EC, rule=initInStg_rule)
#    
#    def initOutStg_rule(model, h, stg, e):
#        return (model.OutStg[h,1,stg,e] == 0)
#    model.initOutStg = Constraint(model.hubs, model.Stg, model.EC, rule=initOutStg_rule)
    
    # storage max charging
    def storageChargeRate_rule(model, h, m, stg, e):
        return (model.InStg[h,m,stg,e] <= model.maxStorCh[stg] * model.CapStg[h,stg,e])
    model.storageChargeRate = Constraint(model.hubs, model.Time, model.Stg, model.EC, rule=storageChargeRate_rule) #maximum charging
    
    # storage max discharging
    def storageDischRate_rule(model, h, m, stg, e):
        return (model.OutStg[h,m,stg,e] <= model.maxStorDisch[stg] * model.CapStg[h,stg,e])
    model.storageDischRate = Constraint(model.hubs,model.Time, model.Stg, model.EC, rule=storageDischRate_rule) #maximum discharging
    
    # storage minimum SoC
    def storageMinState_rule(model, h, m, stg, e):
        return (model.SoC[h,m,stg,e] >= model.CapStg[h,stg,e]*model.minSoC[stg])
    model.storageMinState = Constraint(model.hubs, model.Time, model.Stg, model.EC, rule=storageMinState_rule) #minimum SoC allowed
    
    # SoC <= installed capacity
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