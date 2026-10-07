"""Validate native Physika DFN against a tightly solved PyBaMM oracle."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import resource
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from tutorials.battery.dfn_common import load_native, native_voltage, reference
from tutorials.battery.run_dfn import dfn_options, solve_dfn


def compare(mesh, precision, case, verbose=False):
    initial = None
    if case == 'charge':
        initial = {
            'initial_n': 15522.211404441303,
            'initial_p': 35269.45346021563
        }
    if case == 'protocol':
        initial = {
            'initial_n': 15522.211404441303,
            'initial_p': 35269.45346021563
        }
    current = {
        'discharge': 5.,
        'rest': 0.,
        'charge': -2.5,
        'protocol': 5.
    }[case]
    duration = {
        'discharge': 4000.,
        'rest': 300.,
        'charge': 1200.,
        'protocol': 600.
    }[case]
    model = load_native(mesh, precision, initial, current)
    times = np.arange(0, duration + 1, 50.)
    breaks = []
    if case == 'protocol':
        # Solve each smooth reference segment separately, keeping the state.
        # A PyBaMM time-dependent piecewise current expression uses sign(t).
        import pybamm
        current = 5. * (pybamm.t < 300) - 2.5 * (pybamm.t >= 400)
        model[
            'current_a'] = lambda t: 5. if t < 300 else 0. if t < 400 else -2.5
        breaks = [300., 400.]
    else:
        model['current_a'] = lambda t: float(current)

    def progress(t, stats):
        if verbose and stats.accepted_steps % 100 == 0:
            print(
                f"t={t:.3f}s steps={stats.accepted_steps} retries={stats.rejected_steps}",
                file=sys.stderr,
                flush=True)

    options = dfn_options(progress=progress if verbose else None)
    solution = solve_dfn(model, times, options, breaks)
    # PyBaMM reports the left limit at an exact current discontinuity.
    # Query just to its right for native post-jump states; concentration
    # evolution over 10 microseconds is negligible compared to the gates.
    nt = solution.times.cpu().numpy()
    query_times = nt.copy()
    for jump in breaks:
        indices = np.nonzero(nt == jump)[0]
        if len(indices) > 1:
            query_times[indices[-1]] = jump + 1e-5
    unique_times = np.unique(query_times)
    oracle = reference(mesh, unique_times, current, initial)
    ot = oracle['times']
    xx = solution.differential.cpu().numpy()
    vv = native_voltage(model, solution).cpu().numpy()
    shared = []
    for k, t in enumerate(query_times):
        idx = int(np.argmin(abs(ot - t)))
        if abs(ot[idx] - t) < 1e-8:
            shared.append((k, idx))
    ni, oi = np.array(shared).T
    voltage_error = float(np.max(abs(vv[ni] - oracle['voltage'][oi])))
    concentration_scale = model['x_scale'].cpu().numpy()[:-1]
    concentration_error = float(
        np.max(
            abs(xx[ni, :-1] - oracle['differential'][oi, :-1]) /
            concentration_scale))
    inventory = solution.differential.double(
    ) @ model['inventory_weights'].double()
    drift = float(torch.max(abs(inventory - inventory[0])) / abs(inventory[0]))
    cutoff_error = None
    if case == 'discharge':
        full_oracle = reference(mesh, times, current, initial)
        cutoff_error = abs(
            float(solution.times[-1]) -
            float(full_oracle['times'][-1])) / float(full_oracle['times'][-1])
    metrics = {
        'mesh':
        mesh,
        'precision':
        precision,
        'case':
        case,
        'voltage_error_V':
        voltage_error,
        'concentration_error_scaled':
        concentration_error,
        'inventory_drift_relative':
        drift,
        'cutoff_time_error_relative':
        cutoff_error,
        'termination_time_s':
        float(solution.times[-1]),
        'status':
        solution.status,
        'stats':
        asdict(solution.stats),
        'pybamm_version':
        oracle['pybamm_version'],
        'solver_options': {
            k: v
            for k, v in asdict(options).items() if not callable(v)
        },
        'process_peak_memory_bytes':
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss *
        (1 if sys.platform == 'darwin' else 1024)
    }
    metrics['passed'] = (voltage_error <= 1e-3 and concentration_error <= .01
                         and drift <= 1e-4
                         and (cutoff_error is None or cutoff_error <= .001)
                         and solution.stats.max_algebraic_residual
                         <= options.algebraic_tol * 1.01)
    return metrics


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('--mesh', type=int, default=5)
    cli.add_argument('--precision',
                     choices=['float32', 'float64'],
                     default='float32')
    cli.add_argument('--case',
                     choices=['discharge', 'rest', 'charge', 'protocol'],
                     default='discharge')
    cli.add_argument('--output', type=Path)
    cli.add_argument('--verbose', action='store_true')
    args = cli.parse_args()
    torch.set_num_threads(1)
    result = compare(args.mesh, args.precision, args.case, args.verbose)
    payload = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, flush=True)
    if not result['passed']: raise SystemExit(1)


if __name__ == '__main__': main()
