"""Adaptive BDF2 time integration for the fixed DFN demonstration.

Physika supplies the equations and scales. Torch supplies the Newton
Jacobians and dense linear solves. Backward Euler starts the BDF2 history;
step doubling controls concentration error and implicit re-solves locate
2.5 V cutoff. Accepted states are detached: this is a forward solve.
"""

from dataclasses import dataclass
import math
import time

import torch

# Settings for this demonstration, edited here rather than through a CLI.
RTOL = 1e-5
ATOL = 1e-7
NEWTON_TOL = 5e-7
ALGEBRAIC_TOL = 2e-3
INITIAL_STEP = 0.1
MAX_STEP = 50.0
EVENT_TIME_TOL = 1e-5


class DAEError(RuntimeError):
    """A time step or consistent initialization failed to converge."""


@dataclass
class Solution:
    times: torch.Tensor
    differential: torch.Tensor
    algebraic: torch.Tensor
    reached_cutoff: bool
    stats: dict


def norm(value):
    return float(value.detach().abs().max())


def newton(fun, initial, scales, stats, differential_size=0):
    u = (initial / scales).detach()
    for iteration in range(21):
        residual = fun(u * scales)
        residual_norm = norm(residual)
        if not math.isfinite(residual_norm):
            raise DAEError("Nonfinite nonlinear residual")
        if residual_norm <= NEWTON_TOL:
            return (u * scales).detach()
        if iteration == 20:
            break
        jacobian = torch.autograd.functional.jacobian(
            lambda v: fun(v * scales), u, vectorize=True
        )
        stats["newton_iterations"] += 1
        try:
            delta = torch.linalg.solve(jacobian, -residual).detach()
        except RuntimeError as exc:
            raise DAEError("Newton linear solve failed") from exc
        if not torch.isfinite(delta).all():
            raise DAEError("Nonfinite Newton correction")
        if differential_size:
            # Differential defects can amplify state roundoff. Check the
            # remaining correction tangent to the satisfied constraints.
            correction = delta[:differential_size]
            constraint_norm = norm(residual[differential_size:])
            if constraint_norm <= NEWTON_TOL and norm(correction) > NEWTON_TOL:
                defect = residual.detach().clone()
                defect[differential_size:] = 0
                correction = torch.linalg.solve(jacobian, -defect).detach()[
                    :differential_size
                ]
            if norm(correction) <= NEWTON_TOL and constraint_norm <= NEWTON_TOL:
                return (u * scales).detach()
            alpha = 1.0
        else:
            # Consistent initial potentials need a damped Newton solve.
            alpha = 1.0
            for _ in range(16):
                trial_norm = norm(fun((u + alpha * delta) * scales))
                if math.isfinite(trial_norm) and trial_norm < residual_norm:
                    break
                alpha *= 0.5
            else:
                raise DAEError("Initialization line search failed")
        u = (u + alpha * delta).detach()
    raise DAEError(f"Newton residual {residual_norm:.3g} exceeds tolerance")


def integrate(model, times):
    """Integrate the constant-current model at the requested output times."""
    started = time.perf_counter()
    f, g, cutoff = model["dfn_rhs"], model["dfn_algebraic"], model["dfn_cutoff"]
    x, z = model["x0"].clone(), model["z0"].clone()
    sx, sz, sg = model["x_scale"], model["z_scale"], model["g_scale"]
    nd = len(x)
    scales = torch.cat((sx, sz))
    algebraic_weight = NEWTON_TOL / ALGEBRAIC_TOL
    stats = dict(
        accepted_steps=0,
        rejected_steps=0,
        newton_iterations=0,
        max_algebraic_residual=0.0,
    )
    t = float(times[0])
    z = newton(lambda b: g(t, x, b) / sg * algebraic_weight, z, sz, stats)

    def step(t0, a, b, h, previous=None, previous_h=None):
        if previous is None:
            alpha, history_difference = 1.0, torch.zeros_like(a)
        else:
            ratio = h / previous_h
            alpha = (1 + 2 * ratio) / (1 + ratio)
            history_difference = ratio**2 / (1 + ratio) * (previous - a)

        def residual(y):
            increment, zz = y[:nd], y[nd:]
            xx = a + increment
            return torch.cat(
                (
                    (alpha * increment + history_difference - h * f(t0 + h, xx, zz))
                    / sx,
                    g(t0 + h, xx, zz) / sg * algebraic_weight,
                )
            )

        # Solve for increments instead of subtracting large concentrations.
        y = newton(residual, torch.cat((torch.zeros_like(a), b)), scales, stats, nd)
        return a + y[:nd], y[nd:]

    out_t, out_x, out_z = [], [], []

    def save():
        constraint_norm = norm(g(t, x, z) / sg)
        stats["max_algebraic_residual"] = max(
            stats["max_algebraic_residual"], constraint_norm
        )
        if constraint_norm > ALGEBRAIC_TOL * 1.01:
            raise DAEError("Saved state violates the algebraic tolerance")
        out_t.append(t)
        out_x.append(x.detach().clone())
        out_z.append(z.detach().clone())

    save()
    margin = float(cutoff(t, x, z))
    h = INITIAL_STEP
    previous, previous_h = None, None
    reached_cutoff = False
    for stop in times[1:]:
        stop = float(stop)
        retries = 0
        while t < stop - 1e-12 * max(1.0, abs(stop)):
            if stats["accepted_steps"] >= 10000:
                raise DAEError("Maximum number of time steps exceeded")
            hh = min(h, stop - t)
            if hh < 1e-9 or t + hh == t:
                raise DAEError(f"Time step too small at t={t:.12g}")
            try:
                a, b = step(t, x, z, hh, previous, previous_h)
                half_a, half_b = step(t, x, z, hh / 2, previous, previous_h)
                small_a, _ = step(t + hh / 2, half_a, half_b, hh / 2, x, hh / 2)
                weight = ATOL + RTOL * torch.maximum((a / sx).abs(), (x / sx).abs())
                order = 1 if previous is None else 2
                error = float(
                    torch.sqrt(
                        torch.mean(
                            ((a - small_a) / sx / ((2**order - 1) * weight)) ** 2
                        )
                    )
                )
                if not math.isfinite(error) or error > 1:
                    raise DAEError("Local error test failed")
            except DAEError as exc:
                stats["rejected_steps"] += 1
                retries += 1
                if retries > 15:
                    raise DAEError(f"Integration failed at t={t:.12g}: {exc}") from exc
                h = hh / 2
                continue
            retries = 0
            new_t = t + hh
            new_margin = float(cutoff(new_t, a, b))
            if margin > 0 and new_margin <= 0:
                lo, hi = 0.0, hh
                while hi - lo > EVENT_TIME_TOL:
                    mid = (lo + hi) / 2
                    ma, mb = step(t, x, z, mid, previous, previous_h)
                    if float(cutoff(t + mid, ma, mb)) <= 0:
                        hi, a, b = mid, ma, mb
                    else:
                        lo = mid
                new_t, reached_cutoff = t + hi, True
            previous, previous_h = x, hh
            t, x, z, margin = new_t, a, b, new_margin
            stats["accepted_steps"] += 1
            if reached_cutoff:
                break
            factor = (
                2.0 if error < 1e-12 else max(0.5, min(2.0, 0.9 * error ** (-1 / 3)))
            )
            h = min(MAX_STEP, hh * factor)
        save()
        if reached_cutoff:
            break
    stats["elapsed_seconds"] = time.perf_counter() - started
    return Solution(
        torch.tensor(out_t),
        torch.stack(out_x),
        torch.stack(out_z),
        reached_cutoff,
        stats,
    )
