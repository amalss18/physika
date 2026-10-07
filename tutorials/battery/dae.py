"""Forward integration of semi-explicit index-one DAEs using Torch.

Callbacks evaluate dimensional equations; scaling is applied only inside the
nonlinear solve. Accepted states are detached. This is a forward solver, not
an implementation of implicit sensitivities or a trajectory adjoint.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional
import math
import time

import torch

Residual = Callable[[float, torch.Tensor, torch.Tensor], torch.Tensor]


class DAEError(RuntimeError):
    """A DAE solve could not produce a converged, consistent state."""


@dataclass(frozen=True)
class DAEOptions:
    method: str = "bdf2"
    rtol: float = 1e-4
    atol: float = 1e-6
    initial_step: float = 0.1
    max_step: float = 100.0
    min_step: float = 1e-9
    newton_tol: float = 1e-6
    algebraic_tol: Optional[float] = None
    max_newton_iterations: int = 20
    max_retries: int = 15
    max_steps: int = 100000
    adaptive: bool = True
    store_internal: bool = False
    suppress_algebraic_error: bool = False
    event_time_tol: float = 1e-5
    # These interfaces allow changing Jacobian/linear backends independently.
    jacobian: Optional[Callable] = None
    linear_solve: Optional[Callable] = None
    progress: Optional[Callable] = None


@dataclass
class DAEStats:
    accepted_steps: int = 0
    rejected_steps: int = 0
    newton_iterations: int = 0
    jacobian_evaluations: int = 0
    residual_evaluations: int = 0
    elapsed_seconds: float = 0.0
    max_algebraic_residual: float = 0.0
    rejection_reasons: dict = field(default_factory=dict)
    last_failure: str = ""


@dataclass
class DAESolution:
    times: torch.Tensor
    differential: torch.Tensor
    algebraic: torch.Tensor
    status: int  # 0: final time, 1: terminal event
    event_index: int
    stats: DAEStats
    internal_times: torch.Tensor = field(
        default_factory=lambda: torch.empty(0))
    internal_differential: Optional[torch.Tensor] = None
    internal_algebraic: Optional[torch.Tensor] = None


def dae_options(method=2,
                rtol=1e-4,
                atol=1e-6,
                initial_step=0.1,
                max_step=100.0,
                newton_tol=1e-6,
                algebraic_tol=None,
                suppress_algebraic_error=0,
                store_internal=0):
    """Numeric convenience constructor: method 1 is BE, 2 is BDF2.

    This is a tutorial Python helper, not a Physika language primitive.
    """
    if float(method) != int(method) or int(method) not in (1, 2):
        raise ValueError("DAE method must be 1 (backward Euler) or 2 (BDF2)")
    return DAEOptions(
        method="be" if int(method) == 1 else "bdf2",
        rtol=float(rtol),
        atol=float(atol),
        initial_step=float(initial_step),
        max_step=float(max_step),
        newton_tol=float(newton_tol),
        algebraic_tol=None if algebraic_tol is None else float(algebraic_tol),
        suppress_algebraic_error=bool(suppress_algebraic_error),
        store_internal=bool(store_internal))


def _jacobian(fun, value):
    # autograd.functional supports ordinary compiled callbacks with local
    # tensor writes; vectorized reverse sweeps avoid one Python pass per row.
    return torch.autograd.functional.jacobian(fun, value, vectorize=True)


def _norm(value):
    return float(value.detach().abs().max()) if value.numel() else 0.0


def _newton(fun,
            initial,
            scales,
            options,
            stats,
            line_search=False,
            constraint_start=None):
    u = (initial / scales).detach()
    jacobian = options.jacobian or _jacobian
    linear_solve = options.linear_solve or torch.linalg.solve
    for iteration in range(options.max_newton_iterations + 1):
        r = fun(u * scales)
        stats.residual_evaluations += 1
        norm = _norm(r)
        if not math.isfinite(norm):
            raise DAEError("Nonfinite nonlinear residual")
        if norm <= options.newton_tol:
            return (u * scales).detach()
        if iteration == options.max_newton_iterations:
            break
        with torch.enable_grad():
            j = jacobian(lambda v: fun(v * scales), u)
        stats.jacobian_evaluations += 1
        stats.newton_iterations += 1
        if not torch.isfinite(j).all():
            raise DAEError("Nonfinite nonlinear Jacobian")
        try:
            delta = linear_solve(j, -r).detach()
        except RuntimeError as exc:
            raise DAEError("Singular or failed Newton linear solve") from exc
        if not torch.isfinite(delta).all():
            raise DAEError("Nonfinite Newton correction")
        # Stiff differential defects amplify floating-point state noise.
        # Like IDA, also test the scaled state correction. Algebraic
        # constraints must still meet their explicit residual tolerance.
        correction = (delta[:constraint_start] if constraint_start is not None
                      and options.suppress_algebraic_error else delta)
        if (constraint_start is not None and options.suppress_algebraic_error
                and r[constraint_start:].numel()
                and _norm(r[constraint_start:]) <= options.newton_tol
                and _norm(correction) > options.newton_tol):
            # Constraints already satisfy their tolerance. Estimate the
            # remaining differential correction tangent to that constraint
            # manifold, rather than repeatedly chasing constraint roundoff.
            differential_r = r.detach().clone()
            differential_r[constraint_start:] = 0
            correction = linear_solve(
                j, -differential_r).detach()[:constraint_start]
        if (constraint_start is not None
                and _norm(correction) <= options.newton_tol
                and _norm(r[constraint_start:]) <= options.newton_tol):
            return (u * scales).detach()
        alpha = 1.0
        if line_search:
            for _ in range(16):
                trial = fun((u + alpha * delta) * scales)
                stats.residual_evaluations += 1
                trial_norm = _norm(trial)
                if math.isfinite(trial_norm) and trial_norm < norm:
                    break
                alpha *= 0.5
            else:
                raise DAEError("Initialization line search failed")
        u = (u + alpha * delta).detach()
    index = int(r.detach().abs().argmax())
    raise DAEError(f"Newton failed: scaled residual {norm:.3g} at component "
                   f"{index} exceeds {options.newton_tol:.3g}")


def dae_solve(f: Residual,
              g: Residual,
              x0,
              z0,
              times,
              x_scale,
              z_scale,
              g_scale,
              options=None,
              breakpoints=(),
              events=None):
    """Solve x'=f(t,x,z), g(t,x,z)=0 with prescribed initial x.

    State/scaling inputs are one-dimensional. ``times`` are strictly
    increasing requested outputs; breakpoints are known discontinuities.
    Both sides of a breakpoint are retained with the same timestamp.
    Event callbacks return a vector: crossing from positive to nonpositive
    terminates integration. No clipping/projection of conserved states occurs.
    Absolute time-error tolerances apply to states divided by their scales.
    """
    started = time.perf_counter()
    options = options or DAEOptions()
    if options.method not in ("be", "bdf2"):
        raise ValueError("Unknown DAE integration method")
    for name in ("rtol", "atol", "initial_step", "min_step", "max_step",
                 "newton_tol", "event_time_tol"):
        if not math.isfinite(getattr(options, name)) or getattr(options,
                                                                name) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    x = torch.as_tensor(x0).detach().clone()
    if x.is_complex():
        raise ValueError("DAE states must be real")
    if not x.is_floating_point():
        x = x.to(torch.get_default_dtype())
    dtype, device = x.dtype, x.device
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("DAE states must be float32 or float64")

    def tensor(v):
        if torch.as_tensor(v).is_complex():
            raise ValueError("DAE states and scales must be real")
        return torch.as_tensor(v, dtype=dtype, device=device).detach().clone()

    z = tensor(z0)
    sx, sz, sg = tensor(x_scale), tensor(z_scale), tensor(g_scale)
    for name, v, ref in (("x_scale", sx, x), ("z_scale", sz, z), ("g_scale",
                                                                  sg, z)):
        if v.ndim != 1 or v.shape != ref.shape or not torch.isfinite(
                v).all() or (v <= 0).any():
            raise ValueError(
                f"{name} must be a finite positive vector matching its state")
    if x.ndim != 1 or z.ndim != 1 or not torch.isfinite(
            x).all() or not torch.isfinite(z).all():
        raise ValueError(
            "Initial states must be finite one-dimensional vectors")
    # Keep time arithmetic in double even for single-precision state vectors.
    if torch.as_tensor(times).is_complex():
        raise ValueError("DAE times must be real")
    requested = torch.as_tensor(times, dtype=torch.float64).cpu().tolist()
    if len(requested) < 2 or any(
            not math.isfinite(t) for t in requested) or any(
                b <= a for a, b in zip(requested, requested[1:])):
        raise ValueError("Output times must be finite and strictly increasing")
    breaks = sorted(set(float(v) for v in breakpoints))
    if any(not math.isfinite(v) or not requested[0] < v < requested[-1]
           for v in breaks):
        raise ValueError("Breakpoints must be inside the integration interval")
    stats = DAEStats()
    algebraic_tol = options.newton_tol if options.algebraic_tol is None else options.algebraic_tol
    if not math.isfinite(algebraic_tol) or algebraic_tol <= 0:
        raise ValueError("algebraic_tol must be finite and positive")
    algebraic_weight = options.newton_tol / algebraic_tol

    def call(cb, t, a, b, shape):
        out = cb(t, a, b)
        if not isinstance(out, torch.Tensor) or out.shape != shape:
            raise ValueError("DAE callback returned an incorrect tensor shape")
        if out.dtype != dtype or out.device != device:
            raise ValueError(
                "DAE callback must preserve the state dtype and device")
        return out

    def rhs(t, a, b):
        return call(f, t, a, b, x.shape)

    def alg(t, a, b):
        return call(g, t, a, b, z.shape)

    def consistent(t, a, guess):
        return _newton(lambda b: alg(t, a, b) / sg * algebraic_weight,
                       guess,
                       sz,
                       options,
                       stats,
                       line_search=True)

    t = requested[0]
    # Validate f even if the requested window would immediately terminate.
    rhs(t, x, z)
    z = consistent(t, x, z)
    all_scales = torch.cat((sx, sz))

    def step(t0, a, b, h, previous=None, previous_h=None):
        tn = t0 + h
        if previous is None or options.method == "be":
            alpha = 1.0
            history_difference = torch.zeros_like(a)
        else:
            ratio = h / previous_h
            alpha = (1 + 2 * ratio) / (1 + ratio)
            history_difference = ratio**2 / (1 + ratio) * (previous - a)
        # State-scaled defect is preferable to derivative scaling for tiny h.
        def residual(y):
            increment, zz = y[:x.numel()], y[x.numel():]
            xx = a + increment
            return torch.cat(
                ((alpha * increment + history_difference - h * rhs(tn, xx, zz))
                 / sx, alg(tn, xx, zz) / sg * algebraic_weight))

        # Solve for concentration increments, avoiding cancellation between
        # O(c_max) states when a time step changes concentration by O(1).
        guess = torch.cat((torch.zeros_like(a), b))
        y = _newton(residual,
                    guess,
                    all_scales,
                    options,
                    stats,
                    constraint_start=x.numel())
        return a + y[:x.numel()], y[x.numel():]

    def event_values(tt, a, b):
        if events is None:
            return None
        values = events(tt, a, b).detach()
        if values.ndim != 1 or not torch.isfinite(values).all():
            raise DAEError("Events must return a finite vector")
        return values

    out_t, out_x, out_z = [], [], []
    internal_t, internal_x, internal_z = [], [], []

    def save(tt, a, b, internal=False):
        norm = _norm(alg(tt, a, b) / sg)
        stats.max_algebraic_residual = max(stats.max_algebraic_residual, norm)
        if norm > algebraic_tol * 1.01:
            raise DAEError("Saved state violates the algebraic tolerance")
        target = (internal_t, internal_x,
                  internal_z) if internal else (out_t, out_x, out_z)
        target[0].append(tt)
        target[1].append(a.detach().clone())
        target[2].append(b.detach().clone())

    save(t, x, z)
    ev = event_values(t, x, z)
    status, event_index = 0, -1
    if ev is not None and (ev <= 0).any():
        status, event_index = 1, int(torch.nonzero(ev <= 0)[0])
    h = min(options.initial_step, options.max_step)
    previous, previous_h = None, None
    stops = sorted(set(requested[1:] + breaks))
    for stop in stops:
        if status:
            break
        is_break = stop in breaks
        # Evaluate the left limit at discontinuities; no step straddles a jump.
        integration_stop = math.nextafter(stop,
                                          -math.inf) if is_break else stop
        retries = 0
        while t < integration_stop - 1e-12 * max(1.0, abs(stop)):
            if stats.accepted_steps >= options.max_steps:
                raise DAEError("Maximum number of DAE steps exceeded")
            hh = min(h, integration_stop - t)
            if hh < options.min_step or t + hh == t:
                raise DAEError(f"DAE step too small at t={t:.12g}")
            try:
                a, b = step(t, x, z, hh, previous, previous_h)
                error = 0.0
                if options.adaptive:
                    # Step doubling estimates LTE for both differential and
                    # algebraic states. The full-step solution is accepted,
                    # preserving the chosen BE/BDF2 discretization.
                    half_a, half_b = step(t, x, z, hh / 2, previous,
                                          previous_h)
                    small_a, small_b = step(
                        t + hh / 2, half_a, half_b, hh / 2,
                        x if options.method == "bdf2" else None, hh / 2)
                    weight = options.atol + options.rtol * torch.maximum(
                        torch.cat((a / sx, b / sz)).abs(),
                        torch.cat((x / sx, z / sz)).abs())
                    defect = torch.cat(
                        ((a - small_a) / sx, (b - small_b) / sz))
                    if options.suppress_algebraic_error:
                        defect, weight = defect[:x.numel()], weight[:x.numel()]
                    order = 1 if previous is None or options.method == "be" else 2
                    error = float(
                        torch.sqrt(
                            torch.mean(
                                (defect / ((2**order - 1) * weight))**2)))
                    if not math.isfinite(error) or error > 1:
                        raise DAEError("Local error test failed")
            except DAEError as exc:
                stats.rejected_steps += 1
                stats.last_failure = str(exc)
                reason = str(exc).split(":")[0]
                stats.rejection_reasons[reason] = stats.rejection_reasons.get(
                    reason, 0) + 1
                retries += 1
                if retries > options.max_retries:
                    raise DAEError(f"DAE failed at t={t:.12g}: {exc}") from exc
                h = hh * 0.5
                continue
            retries = 0
            new_t = t + hh
            new_ev = event_values(new_t, a, b)
            if ev is not None and ((ev > 0) & (new_ev <= 0)).any():
                crossings = torch.nonzero((ev > 0)
                                          & (new_ev <= 0)).flatten().tolist()
                roots = []
                for idx in crossings:
                    lo, hi = 0.0, hh
                    ra, rb = a, b
                    while hi - lo > options.event_time_tol:
                        mid = (lo + hi) / 2
                        ma, mb = step(t, x, z, mid, previous, previous_h)
                        if event_values(t + mid, ma, mb)[idx] <= 0:
                            hi, ra, rb = mid, ma, mb
                        else:
                            lo = mid
                    roots.append((hi, idx, ra, rb))
                dh, event_index, a, b = min(roots, key=lambda v: v[0])
                new_t, status = t + dh, 1
            previous, previous_h = x, hh
            t, x, z, ev = new_t, a, b, new_ev
            stats.accepted_steps += 1
            if options.progress is not None:
                options.progress(t, stats)
            if options.store_internal:
                save(t, x, z, internal=True)
            if status:
                save(t, x, z)
                break
            order = 1 if options.method == "be" else 2
            factor = 2.0 if error < 1e-12 else max(
                0.5, min(2.0, 0.9 * error**(-1 / (order + 1))))
            h = min(options.max_step, hh * factor)
        if status:
            break
        t = stop
        if is_break:
            # Use left-limit evaluation when recording the pre-jump state.
            save(math.nextafter(stop, -math.inf), x, z)
            out_t[-1] = stop
            z = consistent(t, x, z)
            save(t, x, z)
            previous, previous_h = None, None
            h = min(options.initial_step, options.max_step)
            ev = event_values(t, x, z)
            if ev is not None and (ev <= 0).any():
                status, event_index = 1, int(torch.nonzero(ev <= 0)[0])
        else:
            save(t, x, z)
    stats.elapsed_seconds = time.perf_counter() - started
    return DAESolution(
        torch.tensor(out_t, dtype=torch.float64, device=device),
        torch.stack(out_x), torch.stack(out_z), status, event_index, stats,
        torch.tensor(internal_t, dtype=torch.float64, device=device),
        torch.stack(internal_x) if internal_x else None,
        torch.stack(internal_z) if internal_z else None)
