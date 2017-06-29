"""
Provides a class for encapsulating an energy hub model.
"""
import functools
from typing import List, Tuple

from pyomo.core.base import (
    ConcreteModel, Set, Param, NonNegativeReals, Binary, Var, Reals,
    Constraint, ConstraintList, Objective, minimize,
)
from pyomo.opt import SolverFactory, SolverManagerFactory

# This import is used to find solvers.
# noinspection PyUnresolvedReferences
# pylint: disable=unused-import
import pyomo.environ

import excel_to_request_format
from data_formats import response_format
from energy_hub import Storage
from energy_hub.input_data import InputData
from energy_hub.range_set import RangeSet
from config import SETTINGS

BIG_M = 5000
TIME_HORIZON = 20
MAX_CARBON = 650000
MAX_SOLAR_AREA = 500


def constraint(*args, enabled=True):
    """
    Mark a function as a constraint of the model.

    The function that adds these constraints to the model is
    `_add_constraints_new`.

    Args:
        *args: The arguments that are passed to Pyomo's Constraint constructor
        enabled: Is the constraint enabled? Defaults to True.

    Returns:
        The decorated function
    """
    def _wrapper(func):
        functools.wraps(func)

        func.is_constraint = True
        func.args = args
        func.enabled = enabled

        return func

    return _wrapper


def constraint_list(*, enabled=True):
    """
    Mark a function as a ConstraintList of the model.

    The function has to return a generator. ie: must use a yield in the method
    body.

    Args:
        enabled: Is the constraint enabled? Defaults to True.

    Returns:
        The decorated function
    """
    def _wrapper(func):
        functools.wraps(func)

        func.is_constraint_list = True
        func.enabled = enabled

        return func

    return _wrapper


class EHubModel:
    """
    Represents a black-box Energy Hub.
    """

    def __init__(self, *, excel=None, request=None):
        """Create a new Energy Hub using some input data.

        Args:
            excel: The Excel 2003 file for input data
            request: The request format dictionary
        """
        self._model = None
        self._data = None

        if excel:
            request = excel_to_request_format.convert(excel)
            self._data = InputData(request)

        if request:
            self._data = InputData(request)

        if self._data:
            self._prepare()
        else:
            raise RuntimeError("Can't create a hub with no data.")

    def _prepare(self):
        self._model = ConcreteModel()

        self._create_sets()

        self._add_parameters()
        self._add_variables()
        self._add_constraints()
        self._add_objective()

    def _create_sets(self):
        data = self._data
        model = self._model

        num_time_steps, num_demands = data.demand_data.shape
        model.time = RangeSet(stop=num_time_steps)
        model.sub_time = RangeSet(start=1, stop=num_time_steps,
                                  within=model.time)

        model.technologies = RangeSet(stop=len(data.converters))

        model.techs_without_grid = RangeSet(start=1, stop=len(data.converters),
                                            within=model.technologies)

        model.storages = RangeSet(stop=len(data.storages))
        model.energy_carrier = RangeSet(stop=num_demands)
        model.demands = RangeSet(stop=num_demands)

        model.techs = Set(initialize=data.max_capacity.keys(),
                          within=model.technologies)
        model.solar_techs = Set(initialize=data.solar_techs,
                                within=model.technologies)
        model.disp_techs = Set(initialize=data.disp_techs,
                               within=model.technologies)
        model.roof_tech = Set(initialize=data.roof_tech,
                              within=model.technologies)

        model.part_load = Set(initialize=data.part_load_techs,
                              within=model.technologies)

    def _add_objective(self):
        def _rule(model):
            return model.total_cost

        self._model.total_cost_objective = Objective(rule=_rule, sense=minimize)

    @staticmethod
    @constraint()
    def calculate_total_cost(model: ConcreteModel) -> bool:
        """
        A constraint for calculating the total cost.

        Args:
            model: The Pyomo model
        """
        cost = (model.investment_cost
                + model.operating_cost
                + model.maintenance_cost)
        income = model.income_from_exports

        return model.total_cost == cost - income

    @staticmethod
    @constraint()
    def calculate_total_carbon(model):
        """
        A constraint for calculating the total carbon produced.

        Args:
            model: The Pyomo model
        """
        total_carbon = 0
        for tech in model.technologies:
            total_energy_imported = sum(model.energy_imported[t, tech]
                                        for t in model.time)
            carbon_factor = model.CARBON_FACTORS[tech]

            total_carbon += carbon_factor * total_energy_imported

        return model.total_carbon == total_carbon

    @staticmethod
    @constraint()
    def calculate_investment_cost(model):
        """
        A constraint for calculating the investment cost.

        Args:
            model: The Pyomo model
        """
        storage_cost = sum(model.NET_PRESENT_VALUE_STORAGE[storage]
                           * model.LINEAR_STORAGE_COSTS[storage]
                           * model.storage_capacity[storage]
                           for storage in model.storages)

        tech_cost = sum(model.NET_PRESENT_VALUE_TECH[tech]
                        * model.LINEAR_CAPITAL_COSTS[tech, out]
                        * model.capacities[tech, out]
                        # + (model.fixCapCosts[tech,out]
                        # * model.Ytechnologies[tech,out])
                        for tech in model.techs_without_grid
                        for out in model.energy_carrier)

        cost = tech_cost + storage_cost
        return model.investment_cost == cost

    @staticmethod
    @constraint()
    def calculate_income_from_exports(model):
        """
        A constraint for calculating the income from exported streams.

        Args:
            model: The Pyomo model
        """
        income = 0
        for energy in model.energy_carrier:
            total_energy_exported = sum(model.energy_exported[t, energy]
                                        for t in model.time)

            income += model.FEED_IN_TARIFFS[energy] * total_energy_exported

        return model.income_from_exports == income

    @staticmethod
    @constraint()
    def calculate_maintenance_cost(model):
        """
        A constraint for calculating the maintenance cost.

        Args:
            model: The Pyomo model
        """
        cost = 0
        for t in model.time:
            for tech in model.technologies:
                for energy in model.energy_carrier:
                    cost += (model.energy_imported[t, tech]
                             * model.CONVERSION_EFFICIENCY[tech, energy]
                             * model.OMV_COSTS[tech])

        return model.maintenance_cost == cost

    @staticmethod
    @constraint()
    def calculate_operating_cost(model):
        """
        A constraint for calculating the operating cost.

        Args:
            model: The Pyomo model
        """
        cost = 0
        for tech in model.technologies:
            total_energy_imported = sum(model.energy_imported[t, tech]
                                        for t in model.time)

            cost += model.OPERATING_PRICES[tech] * total_energy_imported

        return model.operating_cost == cost

    @staticmethod
    @constraint('time', 'storages')
    def storage_capacity(model, t, storage):
        """
        Ensure the storage level is below the storage's capacity.

        Args:
            model: The Pyomo model
            t: A time step
            storage: A storage
        """
        storage_level = model.storage_level[t, storage]
        storage_capacity = model.storage_capacity[storage]

        return storage_level <= storage_capacity

    @staticmethod
    @constraint('time', 'storages')
    def storage_min_state(model, t, storage):
        """
        Ensure the storage level is above it's minimum level.\

        Args:
            model: The Pyomo model
            t: A time step
            storage: A storage
        """
        storage_capacity = model.storage_capacity[storage]
        min_soc = model.MIN_STATE_OF_CHARGE[storage]
        storage_level = model.storage_level[t, storage]

        min_storage_level = storage_capacity * min_soc

        return min_storage_level <= storage_level

    @staticmethod
    @constraint('time', 'storages')
    def storage_discharge_rate(model, t, storage):
        """
        Ensure the discharge rate of a storage is below it's maximum rate.

        Args:
            model: The Pyomo model
            t: A time step
            storage: A storage
        """
        max_discharge_rate = model.MAX_DISCHARGE_RATE[storage]
        storage_capacity = model.storage_capacity[storage]
        discharge_rate = model.energy_from_storage[t, storage]

        max_rate = max_discharge_rate * storage_capacity

        return discharge_rate <= max_rate

    @staticmethod
    @constraint('time', 'storages')
    def storage_charge_rate(model, t, storage):
        """
        Ensure the charge rate of a storage is below it's maximum rate.

        Args:
            model: The Pyomo model
            t: A time step
            storage: A storage
        """
        max_charge_rate = model.MAX_CHARGE_RATE[storage]
        storage_capacity = model.storage_capacity[storage]
        charge_rate = model.energy_to_storage[t, storage]

        max_rate = max_charge_rate * storage_capacity

        return charge_rate <= max_rate

    @staticmethod
    @constraint('sub_time', 'storages')
    def storage_balance(model, t, storage):
        """
        Calculate the current storage level from the previous level.

        Args:
            model: The Pyomo model
            t: A time step
            storage: A storage
        """
        current_storage_level = model.storage_level[t, storage]
        previous_storage_level = model.storage_level[t - 1, storage]

        storage_standing_loss = model.STORAGE_STANDING_LOSSES[storage]

        discharge_rate = model.DISCHARGING_EFFICIENCY[storage]
        charge_rate = model.CHARGING_EFFICIENCY[storage]

        q_in = model.energy_to_storage[t, storage]
        q_out = model.energy_from_storage[t, storage]

        calculated_level = (
            (storage_standing_loss * previous_storage_level)
            + (charge_rate * q_in)
            - ((1 / discharge_rate) * q_out)
        )
        return current_storage_level == calculated_level

    @staticmethod
    @constraint('technologies', 'energy_carrier')
    def fix_cost_constant(model, tech, out):
        """
        Args:
            model: The Pyomo model
            tech: A converter
            out: A storage
        """
        capacity = model.capacities[tech, out]
        rhs = model.BIG_M * model.Ytechnologies[tech, out]
        return capacity <= rhs

    @staticmethod
    @constraint('roof_tech')
    def roof_area(model, roof):
        """
        Ensure the roof techs are taking up less area than there is roof.

        Args:
            model: The Pyomo model
            roof: A roof converter
        """
        roof_area = sum(model.capacities[roof, d]
                        for d in model.energy_carrier)
        max_roof_area = model.MAX_SOLAR_AREA

        return roof_area <= max_roof_area

    @staticmethod
    @constraint('time', 'solar_techs', 'energy_carrier')
    def solar_input(model, t, solar_tech, out):
        """
        Calculate the energy from the roof techs per time step.

        Args:
            model: The Pyomo model
            t: A time step
            solar_tech: A solar converter
            out:
        """
        conversion_rate = model.CONVERSION_EFFICIENCY[solar_tech, out]

        if conversion_rate <= 0:
            return Constraint.Skip

        energy_imported = model.energy_imported[t, solar_tech]
        capacity = model.capacities[solar_tech, out]

        rhs = model.SOLAR_EM[t] * capacity

        return energy_imported == rhs

    @staticmethod
    @constraint('time', 'part_load', 'energy_carrier')
    def part_load_u(model, t, disp, out):
        """
        Args:
            model: The Pyomo model
            t: A time step
            disp: A dispatch tech
            out:
        """
        conversion_rate = model.CONVERSION_EFFICIENCY[disp, out]

        if conversion_rate <= 0:
            return Constraint.Skip

        energy_imported = model.energy_imported[t, disp]

        lhs = energy_imported * conversion_rate
        rhs = model.BIG_M * model.Yon[t, disp]

        return lhs <= rhs

    @staticmethod
    @constraint('time', 'part_load', 'energy_carrier')
    def part_load_l(model, t, disp, out):
        """
        Args:
            model: The Pyomo model
            t: A time step
            disp: A dispatch tech
            out:
        """
        conversion_rate = model.CONVERSION_EFFICIENCY[disp, out]

        if conversion_rate <= 0:
            return Constraint.Skip

        part_load = model.PART_LOAD[disp, out]
        capacity = model.capacities[disp, out]
        energy_imported = model.energy_imported[disp, out]

        lhs = part_load * capacity

        rhs = (energy_imported * conversion_rate
               + model.BIG_M * (1 - model.Yon[t, disp]))
        return lhs <= rhs

    @staticmethod
    @constraint('technologies', 'energy_carrier')
    def capacity(model, tech, out):
        """
        Args:
            model: The Pyomo model
            tech: A converter
            out:
        """
        if model.CONVERSION_EFFICIENCY[tech, out] <= 0:
            return model.capacities[tech, out] == 0

        return Constraint.Skip

    @staticmethod
    @constraint('disp_techs', 'energy_carrier')
    def max_capacity(model, tech, out):
        """
        Args:
            model: The Pyomo model
            tech: A converter
            out:
        """
        return model.capacities[tech, out] <= model.MAX_CAP_TECHS[tech]

    def _get_stream_from_storage(self, storage_index: int) -> int:
        """
        Get the index of the storage's stream.

        Args:
            storage_index: The index of the storage

        Returns:
            The index of the storage's stream

        Raises:
            ValueError: There is no index for the stream's storage
        """
        storage = self._data.storages[storage_index - 1]
        stream_names = [stream.name for stream in self._data.streams]

        try:
            # Pyomo uses 1-based indexing
            return stream_names.index(storage.stream) + 1
        except ValueError:
            raise ValueError(
                f'There is no stream "{storage.stream}" in the streams '
                f'section for "{storage.name}".'
            ) from None

    def _get_storages_from_stream(self, out: int) -> List[Tuple[int, Storage]]:
        stream_name = self._data.time_series_list[out - 1].stream

        return [(i, storage)
                for i, storage in enumerate(self._data.storages, start=1)
                if storage.stream == stream_name]

    @constraint('time', 'demands')
    def loads_balance(self, model, t, demand):
        """
        Args:
            model: The Pyomo model
            t: A time step
            demand: An output stream
        """
        load = model.LOADS[t, demand]
        energy_exported = model.energy_exported[t, demand]

        lhs = load + energy_exported

        total_q_out = 0
        total_q_in = 0
        for i, _ in self._get_storages_from_stream(demand):
            total_q_in += model.energy_to_storage[t, i]
            total_q_out += model.energy_from_storage[t, i]

        energy_in = 0
        for tech in model.technologies:
            energy_imported = model.energy_imported[t, tech]
            conversion_rate = model.CONVERSION_EFFICIENCY[tech, demand]

            energy_in += energy_imported * conversion_rate

        rhs = (total_q_out - total_q_in) + energy_in

        return lhs <= rhs

    @staticmethod
    @constraint('time', 'technologies', 'energy_carrier')
    def capacity_const(model, t, tech, output_type):
        """
        Args:
            model: The Pyomo model
            t: A time step
            tech: A converter
            output_type: An output stream
        """
        conversion_rate = model.CONVERSION_EFFICIENCY[tech, output_type]

        if conversion_rate <= 0:
            return Constraint.Skip

        energy_imported = model.energy_imported[t, tech]
        capacity = model.capacities[tech, output_type]

        energy_in = energy_imported * conversion_rate

        return energy_in <= capacity

    def _add_constraints_new(self):
        """Add all the constraint decorated functions to the model."""
        methods = [getattr(self, method)
                   for method in dir(self)
                   if callable(getattr(self, method))]
        rules = (rule for rule in methods if hasattr(rule, 'is_constraint'))

        for rule in rules:
            if not rule.enabled:
                continue

            name = rule.__name__ + '_constraint'
            args = [getattr(self._model, arg) for arg in rule.args]

            setattr(self._model, name, Constraint(*args, rule=rule))

    def _add_constraint_lists(self):
        methods = [getattr(self, method)
                   for method in dir(self)
                   if callable(getattr(self, method))]
        rules = (rule for rule in methods
                 if hasattr(rule, 'is_constraint_list'))

        for rule in rules:
            if not rule.enabled:
                continue

            name = rule.__name__ + '_constraint_list'

            constraints = ConstraintList()
            for expression in rule():
                constraints.add(expression)

            setattr(self._model, name, constraints)

    @constraint_list()
    def capacity_constraints(self):
        for capacity in self._data.capacities:
            variable = getattr(self._model, capacity.name)

            lower_bound = capacity.lower_bound
            upper_bound = capacity.upper_bound

            yield lower_bound <= variable <= upper_bound

    def _add_constraints(self):
        self._add_constraints_new()
        self._add_constraint_lists()

    @constraint_list()
    def _add_unknown_storage_constraint(self):
        """Ensure that the storage level at the beginning is equal to it's end
        level."""
        data = self._data
        model = self._model

        for i in range(data.demand_data.shape[1]):
            last_entry = model.time.last()

            start_level = model.storage_level[1, i]
            end_level = model.storage_level[last_entry, i]

            yield start_level == end_level

    @constraint_list()
    def _add_various_constraints(self):
        data = self._data
        model = self._model

        dispatch_demands = data.dispatch_demands
        for i, chp in enumerate(data.chp_list):
            dd_1 = dispatch_demands[i, 1]
            dd_0 = dispatch_demands[i, 0]

            rhs = (model.CONVERSION_EFFICIENCY[chp, dd_1]
                   / model.CONVERSION_EFFICIENCY[chp, dd_0]
                   * model.capacities[chp, dd_0])
            yield model.capacities[chp, dd_1] == rhs

            yield (model.Ytechnologies[chp, dd_0]
                   == model.Ytechnologies[chp, dd_1])

            yield (model.capacities[chp, dd_0]
                   <= model.MAX_CAP_TECHS[chp]
                   * model.Ytechnologies[chp, dd_0])

    def _add_capacity_variables(self):
        for capacity in self._data.capacities:
            domain = capacity.domain
            name = capacity.name

            setattr(self._model, name, Var(domain=domain))

    def _add_variables(self):
        model = self._model

        self._add_capacity_variables()

        # Global variables
        model.energy_imported = Var(model.time, model.technologies,
                                    domain=NonNegativeReals)
        model.energy_exported = Var(model.time, model.energy_carrier,
                                    domain=NonNegativeReals)

        model.capacities = Var(model.technologies, model.energy_carrier,
                               domain=NonNegativeReals)

        model.Ytechnologies = Var(model.technologies, model.energy_carrier,
                                  domain=Binary)

        model.Yon = Var(model.time, model.technologies, domain=Binary)

        model.total_cost = Var(domain=Reals)
        model.operating_cost = Var(domain=NonNegativeReals)
        model.maintenance_cost = Var(domain=NonNegativeReals)
        model.income_from_exports = Var(domain=NonNegativeReals)
        model.investment_cost = Var(domain=NonNegativeReals)

        model.total_carbon = Var(domain=Reals)

        # Storage variables
        model.energy_to_storage = Var(model.time, model.storages,
                                      domain=NonNegativeReals)
        model.energy_from_storage = Var(model.time, model.storages,
                                        domain=NonNegativeReals)

        model.storage_level = Var(model.time, model.storages,
                                  domain=NonNegativeReals)

        model.storage_capacity = Var(model.storages,
                                     domain=NonNegativeReals)

    def _add_parameters(self):
        data = self._data
        model = self._model

        # coupling matrix & Technical parameters
        # coupling matrix technology efficiencies
        model.CONVERSION_EFFICIENCY = Param(model.technologies,
                                            model.energy_carrier,
                                            initialize=data.c_matrix)
        model.MAX_CAP_TECHS = Param(model.disp_techs,
                                    initialize=data.max_capacity)

        model.MAX_CHARGE_RATE = Param(model.storages,
                                      initialize=data.storage_charge)
        model.MAX_DISCHARGE_RATE = Param(model.storages,
                                         initialize=data.storage_discharge)
        model.STORAGE_STANDING_LOSSES = Param(model.storages,
                                              initialize=data.storage_loss)
        model.CHARGING_EFFICIENCY = Param(model.storages,
                                          initialize=data.storage_ef_ch)
        model.DISCHARGING_EFFICIENCY = Param(model.storages,
                                             initialize=data.storage_ef_disch)
        model.MIN_STATE_OF_CHARGE = Param(model.storages,
                                          initialize=data.storage_min_soc)
        # PartloadInput
        model.PART_LOAD = Param(model.technologies, model.energy_carrier,
                                initialize=data.part_load)

        # carbon factors
        model.CARBON_FACTORS = Param(model.technologies,
                                     initialize=data.carb_factors)
        model.MAX_CARBON = Param(initialize=MAX_CARBON)

        # Cost parameters
        # Technologies capital costs
        model.LINEAR_CAPITAL_COSTS = Param(model.technologies,
                                           model.energy_carrier,
                                           initialize=data.linear_cost)
        model.LINEAR_STORAGE_COSTS = Param(model.storages,
                                           initialize=data.storage_lin_cost)
        # Operating prices technologies
        model.OPERATING_PRICES = Param(model.technologies,
                                       initialize=data.fuel_price)
        model.FEED_IN_TARIFFS = Param(model.energy_carrier,
                                      initialize=data.feed_in)
        # Maintenance costs
        model.OMV_COSTS = Param(model.technologies,
                                initialize=data.var_maintenance_cost)

        # Declaring Global Parameters
        model.TIME_HORIZON = Param(within=NonNegativeReals,
                                   initialize=TIME_HORIZON)

        model.BIG_M = Param(within=NonNegativeReals, initialize=BIG_M)

        model.INTEREST_RATE = Param(within=NonNegativeReals,
                                    initialize=data.interest_rate)

        model.MAX_SOLAR_AREA = Param(initialize=MAX_SOLAR_AREA)

        # loads
        model.LOADS = Param(model.time, model.demands,
                            initialize=data.demands)
        model.SOLAR_EM = Param(model.time, initialize=data.solar_data)

        model.NET_PRESENT_VALUE_TECH = Param(model.technologies,
                                             domain=NonNegativeReals,
                                             initialize=data.tech_npv)
        model.NET_PRESENT_VALUE_STORAGE = Param(model.storages,
                                                domain=NonNegativeReals,
                                                initialize=data.storage_npv)

    def solve(self):
        """
        Solve the model.

        Returns:
            The results
        """
        if not self._model:
            raise RuntimeError("Can't solve a model with no data.")

        solver = SETTINGS["solver"]["name"]
        options = SETTINGS["solver"]["options"]
        if options is None:
            options = {}

        opt = SolverFactory(solver)
        opt.options = options
        solver_manager = SolverManagerFactory("serial")

        results = solver_manager.solve(self._model, opt=opt, tee=True,
                                       timelimit=None)

        # in order to get the solutions found by the solver
        self._model.solutions.store_to(results)

        return response_format.create_response(results, self._model)
