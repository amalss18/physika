"""Developer-only regeneration of the one fixed PyBaMM reference.

Run from the repository root:
    python tutorials/battery/generate_dfn_reference.py
PyBaMM is not used by the tutorial runner or its tests.
"""
import json
from pathlib import Path
import sys

import numpy as np
import pybamm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tutorials.battery.run_dfn import load_model, REFERENCE, TIMES


def main():
    native = load_model()
    parameters = pybamm.ParameterValues('Chen2020')
    parameters.update({'Current function [A]': native['current_nominal']})
    mesh = dict(x_n=native['nx_n'],
                x_s=native['nx_s'],
                x_p=native['nx_p'],
                r_n=native['nr_n'],
                r_p=native['nr_p'])
    simulation = pybamm.Simulation(pybamm.lithium_ion.BasicDFN(),
                                   parameter_values=parameters,
                                   var_pts=mesh,
                                   solver=pybamm.IDAKLUSolver(rtol=1e-9,
                                                              atol=1e-9))
    solution = simulation.solve(t_eval=TIMES, t_interp=TIMES)
    names = dict(cn='Negative particle concentration [mol.m-3]',
                 cp='Positive particle concentration [mol.m-3]',
                 ce='Electrolyte concentration [mol.m-3]',
                 surface_n='Negative particle surface concentration [mol.m-3]',
                 surface_p='Positive particle surface concentration [mol.m-3]',
                 phi_n='Negative electrode potential [V]',
                 phi_p='Positive electrode potential [V]',
                 phi_e='Electrolyte potential [V]',
                 voltage='Voltage [V]')
    # Processed particle variables are r,x,t; the native states are t,x,r.
    data = {
        key: np.moveaxis(solution[name].entries, -1, 0)
        for key, name in names.items()
    }
    data['cn'] = data['cn'].swapaxes(1, 2)
    data['cp'] = data['cp'].swapaxes(1, 2)
    data['times'] = solution.t
    for key, domain in dict(x_n='negative electrode',
                            x_s='separator',
                            x_p='positive electrode',
                            r_n='negative particle',
                            r_p='positive particle').items():
        data[key] = simulation.mesh[domain].nodes
    REFERENCE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(REFERENCE, **data)
    metadata = dict(
        model='PyBaMM BasicDFN',
        parameters='Chen2020',
        pybamm_version=pybamm.__version__,
        mesh=mesh,
        temperature_K=native['temperature'],
        current_A=native['current_nominal'],
        cutoff_V=2.5,
        initial_concentrations_mol_m3=dict(negative=native['initial_n'],
                                           positive=native['initial_p'],
                                           electrolyte=native['initial_e']),
        rtol=1e-9,
        atol=1e-9,
        solver='IDAKLUSolver',
        termination=solution.termination,
        potential_reference='Negative solid current collector = 0 V',
        particle_axes='time, electrode position, particle radius',
        output_spacing_s=50.0,
        regeneration='python tutorials/battery/generate_dfn_reference.py')
    REFERENCE.with_suffix('.json').write_text(
        json.dumps(metadata, indent=2) + '\n')
    print(REFERENCE)


if __name__ == '__main__':
    main()
