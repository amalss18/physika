"""Run the fixed Physika DFN demonstration and compare its saved reference.

From the repository root: python tutorials/battery/run_dfn.py
Edit the .phyk model and dae.py directly to change the experiment.
"""

import json
from pathlib import Path
import sys

import numpy as np
import torch

from physika.codegen import from_ast_to_torch
from physika.lexer import lexer
from physika.parser import parser, symbol_table
from physika.type_checker import TypeChecker
from physika.utils.ast_utils import build_unified_ast
from physika.utils.import_manager import resolve_imports
from tutorials.battery.dae import integrate

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

MODEL = ROOT / "tutorials/doyle_fuller_newman.phyk"
REFERENCE = ROOT / "tests/reference/chen2020_discharge_5.npz"
OUTPUT = ROOT / "tutorials/battery/output"
TIMES = np.arange(0.0, 4001.0, 50.0)
SNAPSHOTS = (100.0, 1500.0, 3000.0)


def load_model():
    """Compile the unmodified Physika source through the standard compiler."""
    torch.set_default_dtype(torch.float64)
    symbol_table.clear()
    lexer.lexer.lineno = 1
    nodes = resolve_imports(parser.parse(MODEL.read_text(), lexer=lexer), MODEL)
    ast = build_unified_ast(nodes, symbol_table)
    errors = TypeChecker(ast).run()
    if errors:
        raise TypeError("\n".join(errors))
    model = {}
    exec(from_ast_to_torch(ast, print_code=False), model)
    return model


def fields(model, solution):
    """Unpack states for plotting; physical reconstruction stays in Physika."""
    x = solution.differential.numpy()
    n, p, ne = model["ns_n"], model["ns_p"], model["ne"]
    potentials = torch.stack(
        [model["physical_potentials"](z) for z in solution.algebraic]
    ).numpy()
    nn, np_ = model["nx_n"], model["nx_p"]
    return dict(
        cn=x[:, :n].reshape(-1, nn, model["nr_n"]),
        cp=x[:, n : n + p].reshape(-1, np_, model["nr_p"]),
        ce=x[:, n + p : n + p + ne],
        surface_n=torch.stack(
            [model["surface_n"](state) for state in solution.differential]
        ).numpy(),
        surface_p=torch.stack(
            [model["surface_p"](state) for state in solution.differential]
        ).numpy(),
        phi_n=potentials[:, :nn],
        phi_p=potentials[:, nn : nn + np_],
        phi_e=potentials[:, nn + np_ :],
        voltage=torch.stack(
            [
                model["dfn_voltage"](float(t), state, z)
                for t, state, z in zip(
                    solution.times, solution.differential, solution.algebraic
                )
            ]
        ).numpy(),
    )


def compare(model, solution, native, reference):
    """Compare all matching saved times, not just the plotted snapshots.

    The two independently located cutoff states are compared through cutoff
    time, avoiding a spatial comparison at different physical times.
    """
    times = solution.times.numpy()[:-1]
    indices = np.searchsorted(reference["times"], times)
    np.testing.assert_allclose(reference["times"][indices], times, atol=1e-8, rtol=0)
    scales = dict(
        cn=model["cmax_n"],
        cp=model["cmax_p"],
        ce=model["initial_e"],
        surface_n=model["cmax_n"],
        surface_p=model["cmax_p"],
    )
    concentrations = {
        name: float(
            np.max(np.abs(native[name][:-1] - reference[name][indices])) / scale
        )
        for name, scale in scales.items()
    }
    potentials = {
        name: float(np.max(np.abs(native[name][:-1] - reference[name][indices])))
        for name in ("phi_n", "phi_p", "phi_e")
    }
    voltage_error = float(
        np.max(np.abs(native["voltage"][:-1] - reference["voltage"][indices]))
    )
    cutoff_error = (
        abs(float(solution.times[-1]) - reference["times"][-1]) / reference["times"][-1]
    )
    inventory = solution.differential @ model["inventory_weights"]
    drift = float(torch.max(abs(inventory - inventory[0])) / abs(inventory[0]))
    passed = (
        solution.reached_cutoff
        and voltage_error <= 1e-3
        and max(concentrations.values()) <= 0.01
        and max(potentials.values()) <= 1e-3
        and cutoff_error <= 0.001
        and drift <= 1e-4
        and solution.stats["max_algebraic_residual"] <= 2e-3 * 1.01
    )
    return dict(
        passed=bool(passed),
        voltage_error_V=voltage_error,
        concentration_errors_scaled=concentrations,
        potential_errors_V=potentials,
        cutoff_time_error_relative=float(cutoff_error),
        inventory_drift_relative=drift,
        termination_time_s=float(solution.times[-1]),
        stats=solution.stats,
    )


def plot_profiles(solution, native, reference):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    colors = ("#2878b5", "#d98324", "#3a9362")
    styles = [
        Line2D([], [], color="0.25", label="Physika"),
        Line2D(
            [],
            [],
            color="0.25",
            linestyle="--",
            marker="o",
            markersize=3,
            label="PyBaMM reference",
        ),
    ]
    times = solution.times.numpy()
    ni = [
        int(np.flatnonzero(np.isclose(times, t, rtol=0, atol=1e-8))[0])
        for t in SNAPSHOTS
    ]
    ri = [
        int(np.flatnonzero(np.isclose(reference["times"], t, rtol=0, atol=1e-8))[0])
        for t in SNAPSHOTS
    ]

    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    ax.plot(times / 3600, native["voltage"], color=colors[0], label="Physika")
    ax.plot(
        reference["times"] / 3600,
        reference["voltage"],
        "--",
        color=colors[1],
        label="PyBaMM reference",
    )
    ax.set(
        xlabel="Discharge time [h]",
        ylabel="Terminal voltage [V]",
        title="Chen2020 · 5 A discharge · 298.15 K",
    )
    ax.legend()
    fig.savefig(OUTPUT / "voltage.png", dpi=180)
    plt.close(fig)

    x_n, x_p = reference["x_n"] * 1e6, reference["x_p"] * 1e6
    x_e = np.concatenate((reference["x_n"], reference["x_s"], reference["x_p"])) * 1e6
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    for ax, name, x, title in zip(
        axes,
        ("ce", "surface_n", "surface_p"),
        (x_e, x_n, x_p),
        ("Electrolyte", "Negative particle surface", "Positive particle surface"),
    ):
        for k, (i, j, color) in enumerate(zip(ni, ri, colors)):
            ax.plot(x, native[name][i], color=color, label=f"{SNAPSHOTS[k]:g} s")
            ax.plot(x, reference[name][j], "--o", color=color, markersize=3)
        ax.set(
            xlabel="Position across cell [µm]",
            ylabel="Concentration [mol/m³]",
            title=title,
        )
    axes[0].legend()
    fig.legend(handles=styles, loc="outside lower center", ncol=2)
    fig.savefig(OUTPUT / "concentrations.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    for ax, name, x, title, scale, unit in zip(
        axes,
        ("phi_e", "phi_n", "phi_p"),
        (x_e, x_n, x_p),
        ("Electrolyte", "Negative solid", "Positive solid"),
        (1.0, 1e6, 1.0),
        ("V", "µV", "V"),
    ):
        for k, (i, j, color) in enumerate(zip(ni, ri, colors)):
            ax.plot(
                x, scale * native[name][i], color=color, label=f"{SNAPSHOTS[k]:g} s"
            )
            ax.plot(x, scale * reference[name][j], "--o", color=color, markersize=3)
        ax.set(
            xlabel="Position across cell [µm]",
            ylabel=f"Potential [{unit}]",
            title=title,
        )
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    axes[0].legend()
    fig.legend(handles=styles, loc="outside lower center", ncol=2)
    fig.savefig(OUTPUT / "potentials.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
    for ax, name, radius, title in zip(
        axes, ("cn", "cp"), ("r_n", "r_p"), ("Negative particle", "Positive particle")
    ):
        middle = native[name].shape[1] // 2
        for k, (i, j, color) in enumerate(zip(ni, ri, colors)):
            ax.plot(
                reference[radius] * 1e6,
                native[name][i, middle],
                color=color,
                label=f"{SNAPSHOTS[k]:g} s",
            )
            ax.plot(
                reference[radius] * 1e6,
                reference[name][j, middle],
                "--o",
                color=color,
                markersize=3,
            )
        ax.set(
            xlabel="Particle radius [µm]",
            ylabel="Concentration [mol/m³]",
            title=f"{title} · middle electrode cell",
        )
    time_handles, _ = axes[0].get_legend_handles_labels()
    fig.legend(handles=time_handles + styles, loc="outside lower center", ncol=5)
    fig.savefig(OUTPUT / "particles.png", dpi=180)
    plt.close(fig)


def main():
    torch.set_num_threads(1)
    model = load_model()
    solution = integrate(model, TIMES)
    native = fields(model, solution)
    with np.load(REFERENCE) as reference:
        report = compare(model, solution, native, reference)
        OUTPUT.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            OUTPUT / "trajectory.npz",
            times=solution.times.numpy(),
            differential=solution.differential.numpy(),
            algebraic=solution.algebraic.numpy(),
            **native,
        )
        plot_profiles(solution, native, reference)
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"Figures and trajectory: {OUTPUT}")
    if not report["passed"]:
        raise SystemExit("DFN reference comparison failed")


if __name__ == "__main__":
    main()
