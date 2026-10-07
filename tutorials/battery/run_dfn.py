"""Run the Physika BasicDFN equations with a tutorial-local forward solver.

Run from the repository root:
    python tutorials/battery/run_dfn.py --precision float64
PyBaMM is not imported by this entry point.
"""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from tutorials.battery.dae import DAEOptions, dae_solve
from tutorials.battery.dfn_common import load_native, native_voltage


def dfn_options(**overrides):
    """Shared tutorial/validation settings, identical in both precisions."""
    return replace(DAEOptions(rtol=1e-5,
                              atol=1e-7,
                              newton_tol=5e-7,
                              algebraic_tol=2e-3,
                              suppress_algebraic_error=True,
                              initial_step=0.1,
                              max_step=50.,
                              max_steps=10000), **overrides)


def solve_dfn(model, output_times, options=None, breakpoints=()):
    """Solve compiled .phyk callbacks without changing global Torch settings."""
    return dae_solve(model['dfn_rhs'], model['dfn_algebraic'], model['x0'],
                     model['z0'], output_times, model['x_scale'],
                     model['z_scale'], model['g_scale'],
                     options or dfn_options(), breakpoints,
                     model['dfn_events'])


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--precision', choices=('float32', 'float64'),
                     default='float32')
    cli.add_argument('--mesh', nargs='+', type=int, default=[5],
                     help='one uniform size, or nx_n nx_s nx_p nr_n nr_p')
    cli.add_argument('--duration', type=float, default=4000.)
    cli.add_argument('--output-step', type=float, default=100.)
    cli.add_argument('--method', choices=('be', 'bdf2'), default='bdf2')
    cli.add_argument('--current', type=float, default=5.,
                     help='prescribed current in A; positive is discharge')
    cli.add_argument('--initial-n', type=float)
    cli.add_argument('--initial-p', type=float)
    cli.add_argument('--initial-e', type=float)
    cli.add_argument('--breakpoints', nargs='*', type=float, default=[])
    cli.add_argument('--output', type=Path,
                     help='optional NPZ trajectory with voltage and JSON stats')
    args = cli.parse_args(argv)
    if len(args.mesh) not in (1, 5):
        cli.error('--mesh requires either one or five sizes')
    if args.duration <= 0 or args.output_step <= 0:
        cli.error('--duration and --output-step must be positive')
    mesh = args.mesh[0] if len(args.mesh) == 1 else args.mesh
    initial = {k: v for k, v in (
        ('initial_n', args.initial_n), ('initial_p', args.initial_p),
        ('initial_e', args.initial_e)) if v is not None}
    # Dense solves at these sizes are faster with one CPU thread. Restore the
    # caller's setting even when invoked as a Python API or a solve fails.
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        model = load_native(mesh, args.precision, initial, args.current)
        times = torch.arange(0., args.duration, args.output_step,
                              dtype=torch.float64).tolist() + [args.duration]
        solution = solve_dfn(model, times, dfn_options(method=args.method),
                             args.breakpoints)
        voltage = native_voltage(model, solution)
        payload = {'precision': args.precision,
                   'status': solution.status,
                   'event_index': solution.event_index,
                   'termination_time_s': float(solution.times[-1]),
                   'voltage_V': float(voltage[-1]),
                   'stats': asdict(solution.stats)}
        if args.output:
            import numpy as np
            args.output.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.output, times=solution.times.cpu().numpy(),
                                differential=solution.differential.cpu().numpy(),
                                algebraic=solution.algebraic.cpu().numpy(),
                                voltage=voltage.cpu().numpy())
            args.output.with_suffix('.json').write_text(
                json.dumps(payload, indent=2) + '\n')
        print(json.dumps(payload, indent=2), flush=True)
    finally:
        torch.set_num_threads(previous_threads)


if __name__ == '__main__':
    main()
