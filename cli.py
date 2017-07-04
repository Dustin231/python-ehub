"""
The CLI for the EHub Model.

The options for running the model are set in the config.yaml file. See that
file for more information.

Given the config.yaml file, solving the model is easy:

    $ python3.6 cli.py

The program will then print out the output of whatever solver is configured in
the config.yaml file and then the solved variables, parameters, constraints,
etc..
"""
from collections import OrderedDict
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

from config import SETTINGS
from energy_hub import EHubModel


def print_section(section_name: str, solution_section: dict) -> None:
    """
    Print all the attributes with a heading.

    Args:
        section_name: The heading
        solution_section: The dictionary with all the attributes
    """
    half_heading = '=' * 10
    print(f"\n{half_heading} {section_name} {half_heading}")

    attributes = OrderedDict(sorted(solution_section.items()))
    for name, value in attributes.items():
        if isinstance(value, dict):
            value = pd.DataFrame.from_dict(value, orient='index')

            if list(value.columns) == [0]:
                value.columns = [name]

        print(f"\n{name}: \n{value}")


def pretty_print(results: dict) -> None:
    """Print the results in a prettier format.

    Args:
        results: The results dictionary to print
    """
    np.set_printoptions(linewidth=1000, suppress=True)

    version = results['version']
    solver = results['solver']

    print("Version: {}".format(version))

    print("Solver")
    print("time: {}".format(solver['time']))
    print("termination condition: {}".format(solver['termination_condition']))

    print("Solution")
    print_section('Objective', results['solution']['objective'])
    print_section('Sets', results['solution']['sets'])
    print_section('Parameters', results['solution']['parameters'])
    print_section('Variable Parameters', results['solution']['param_or_var'])
    print_section('Variables', results['solution']['variables'])


def main():
    """The main function for the CLI."""
    model = EHubModel(excel=SETTINGS["input_file"])

    results = model.solve()

    pretty_print(results)

    with open(SETTINGS["output_file"], 'w') as file:
        with redirect_stdout(file):
            pretty_print(results)


if __name__ == "__main__":
    main()
