"""Reference/validation helpers; the battery equations are compiled .phyk."""
from pathlib import Path
import re
import torch
from physika.parser import parser, symbol_table
from physika.lexer import lexer
from physika.utils.ast_utils import build_unified_ast
from physika.utils.import_manager import resolve_imports
from physika.type_checker import TypeChecker
from physika.codegen import from_ast_to_torch
from tutorials.battery.precision import as_typed_tensor, lower_precision
from tutorials.battery.tensor_operations import bind_tensor_operations

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / 'tutorials/doyle_fuller_newman.phyk'


def load_native(mesh=5, precision='float64', initial=None, current=5.0):
    """Compile the native equations using the unchanged Physika core.

    mesh may be one integer or (nx_n,nx_s,nx_p,nr_n,nr_p). The generated
    precision is bound into the generated constructors and casts. Neither
    compilation nor later callbacks change Torch's ambient default dtype.
    """
    mesh = (mesh, ) * 5 if isinstance(mesh, int) else tuple(mesh)
    if len(mesh) != 5 or any(int(n) != n or n < 3 for n in mesh):
        raise ValueError('DFN meshes require five integer sizes >= 3')
    source = MODEL.read_text()
    for key, n in zip(('nx_n', 'nx_s', 'nx_p', 'nr_n', 'nr_p'), mesh):
        source = re.sub(rf'{key}: ℕ = 5', f'{key}: ℕ = {n}', source)
    symbol_table.clear()
    lexer.lexer.lineno = 1
    nodes = resolve_imports(parser.parse(source, lexer=lexer), MODEL)
    ast = build_unified_ast(nodes, symbol_table)
    initial = dict(initial or {}, current_nominal=float(current))
    if initial is not None:
        for node in ast['program']:
            if node[0] == 'decl' and node[1] in initial:
                i = ast['program'].index(node)
                ast['program'][i] = (*node[:3], ('num',
                                                 float(initial[node[1]])),
                                     node[-1])
    errors = TypeChecker(ast).run()
    if errors:
        raise TypeError('\n'.join(errors))
    namespace = {'as_typed_tensor': as_typed_tensor}
    code = from_ast_to_torch(ast, print_code=False)
    # Bind before executing the top-level initial/reference calculations,
    # which themselves call OCP and exchange-current functions. Rebinding
    # after execution would change their floating-point operation ordering.
    import ast as python_ast
    tree = python_ast.parse(lower_precision(code, precision))
    program_start = next(i for i, node in enumerate(tree.body)
                         if isinstance(node, python_ast.Assign))
    exec(compile(python_ast.Module(body=tree.body[:program_start], type_ignores=[]),
                 str(MODEL), 'exec'), namespace)
    bind_tensor_operations(namespace)
    exec(compile(python_ast.Module(body=tree.body[program_start:], type_ignores=[]),
                 str(MODEL), 'exec'), namespace)
    return namespace


def native_voltage(model, solution):
    return torch.stack([
        model['dfn_voltage'](float(t), x, z) for t, x, z in zip(
            solution.times, solution.differential, solution.algebraic)
    ])


def physical_potentials(model, z):
    z = z.clone()
    n, p = model['nx_n'], model['nx_p']
    z[..., n:n + p] += model['ref_p']
    z[..., n + p:] += model['ref_e']
    return z


def reference(mesh=5, times=None, current=5., initial=None):
    """Tightly solve the matching BasicDFN; never called by the native model."""
    import numpy as np
    import pybamm
    mesh = (mesh, ) * 5 if isinstance(mesh, int) else tuple(mesh)
    parameter = pybamm.ParameterValues('Chen2020')
    updates = {'Current function [A]': current}
    if initial:
        names = {
            'initial_n':
            'Initial concentration in negative electrode [mol.m-3]',
            'initial_p':
            'Initial concentration in positive electrode [mol.m-3]',
            'initial_e': 'Initial concentration in electrolyte [mol.m-3]'
        }
        updates.update({names[k]: v for k, v in initial.items()})
    parameter.update(updates)
    model = pybamm.lithium_ion.BasicDFN()
    vp = dict(zip(('x_n', 'x_s', 'x_p', 'r_n', 'r_p'), mesh))
    simulation = pybamm.Simulation(model,
                                   parameter_values=parameter,
                                   var_pts=vp,
                                   solver=pybamm.IDAKLUSolver(rtol=1e-9,
                                                              atol=1e-9))
    times = np.asarray(
        times if times is not None else np.linspace(0, 4000, 81))
    sol = simulation.solve(t_eval=times, t_interp=times)
    names = ('Negative particle concentration [mol.m-3]',
             'Positive particle concentration [mol.m-3]',
             'Electrolyte concentration [mol.m-3]', 'Discharge capacity [A.h]')
    # State entries from processed variables are ordered r,x,t. Native
    # particle arrays have r running fastest for each x location.
    diff = np.concatenate(
        [sol[n].entries.reshape(-1, len(sol.t), order='F') for n in names],
        axis=0).T
    voltage = sol['Voltage [V]'].entries.reshape(-1)
    return {
        'times': sol.t,
        'differential': diff,
        'voltage': voltage,
        'termination': sol.termination,
        'pybamm_version': pybamm.__version__,
        'mesh': mesh,
        'rtol': 1e-9,
        'atol': 1e-9
    }
