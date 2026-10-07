import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import print

# === Functions ===
def asinh_scalar(q):
    return torch.log((q + torch.sqrt(((q * q) + 1.0) if isinstance(((q * q) + 1.0), torch.Tensor) else torch.tensor(float(((q * q) + 1.0))))) if isinstance((q + torch.sqrt(((q * q) + 1.0) if isinstance(((q * q) + 1.0), torch.Tensor) else torch.tensor(float(((q * q) + 1.0))))), torch.Tensor) else torch.tensor(float((q + torch.sqrt(((q * q) + 1.0) if isinstance(((q * q) + 1.0), torch.Tensor) else torch.tensor(float(((q * q) + 1.0))))))))

def current_a(t):
    return current_nominal

def reaction_n(t, x, z):
    cs = (S_n @ x[int(0):int(ns_n)])
    ce = x[int((ns_n + ns_p)):int(((ns_n + ns_p) + nx_n))]
    phi_s = z[int(0):int(nx_n)]
    phi_e = z[int((nx_n + nx_p)):int(((nx_n + nx_p) + nx_n))]
    eta = ((((phi_s - phi_e) + u_n_initial) + eta_n_reference) - ocp_n((cs / cmax_n)))
    return ((2.0 * exchange_n(ce, cs)) * sinh(((faraday * eta) / ((2.0 * gas_constant) * temperature))))

def reaction_p(t, x, z):
    cs = (S_p @ x[int(ns_n):int((ns_n + ns_p))])
    ce = x[int((((ns_n + ns_p) + nx_n) + nx_s)):int(((ns_n + ns_p) + ne))]
    phi_s = z[int(nx_n):int((nx_n + nx_p))]
    phi_e = z[int((((nx_n + nx_p) + nx_n) + nx_s)):int(na)]
    eta = ((((phi_s - phi_e) + u_p_initial) + eta_p_reference) - ocp_p((cs / cmax_p)))
    return ((2.0 * exchange_p(ce, cs)) * sinh(((faraday * eta) / ((2.0 * gas_constant) * temperature))))

def electrolyte_current(t, x, z):
    ce = x[int((ns_n + ns_p)):int(((ns_n + ns_p) + ne))]
    phi_e = z[int((nx_n + nx_p)):int(na)]
    conductivity = (A_e @ (transport * electrolyte_conductivity(ce)))
    chi = ((((2.0 * (1.0 - t_plus)) * gas_constant) * temperature) / (faraday * maximum(ce, 0.01)))
    return (conductivity * ((harmonic_edges(chi, H_e, ne) * gradient_values(ce, width, ne)) - gradient_values(phi_e, width, ne)))

def dfn_rhs(t, x, z):
    jn = reaction_n(t, x, z)
    jp = reaction_p(t, x, z)
    ce = x[int((ns_n + ns_p)):int(((ns_n + ns_p) + ne))]
    aj = torch.cat([(a_n * jn), separator_zero, (a_p * jp)])
    flux = (((-harmonic_edges((transport * electrolyte_diffusivity(ce)), H_e, ne)) * gradient_values(ce, width, ne)) + ((t_plus * electrolyte_current(t, x, z)) / faraday))
    dcn = ((K_n @ (x[int(0):int(ns_n)] - initial_n)) + (B_n @ jn))
    dcp = ((K_p @ (x[int(ns_n):int((ns_n + ns_p))] - initial_p)) + (B_p @ jp))
    dce = (((-divergence_values(flux, width, ne)) + (aj / faraday)) / eps)
    return torch.cat([dcn, dcp, dce, torch.stack([torch.as_tensor((current_a(t) / 3600.0))])])

def dfn_algebraic(t, x, z):
    jn = reaction_n(t, x, z)
    jp = reaction_p(t, x, z)
    aj = torch.cat([(a_n * jn), separator_zero, (a_p * jp)])
    phin = z[int(0):int(nx_n)]
    phip = z[int(nx_n):int((nx_n + nx_p))]
    gradn = gradient_values(phin, width_n, nx_n)
    gradn[int(0)] = ((2.0 * phin[int(0)]) / width_n[int(0)])
    isn = ((-sigma_n) * gradn)
    boundary = positive_boundary
    isp = (((-sigma_p) * gradient_values(phip, width_p, nx_p)) + ((current_a(t) / area) * boundary))
    gn = ((length_total ** 2) * (divergence_values(isn, width_n, nx_n) + (a_n * jn)))
    gp = ((length_total ** 2) * (divergence_values(isp, width_p, nx_p) + (a_p * jp)))
    ge = ((length_total ** 2) * (divergence_values(electrolyte_current(t, x, z), width, ne) - aj))
    return torch.cat([gn, gp, ge])

def dfn_voltage(t, x, z):
    return ((ref_p + (1.5 * z[int(((nx_n + nx_p) - 1))])) - (0.5 * z[int(((nx_n + nx_p) - 2))]))

def physical_potentials(z):
    return torch.cat([z[int(0):int(nx_n)], (ref_p + z[int(nx_n):int((nx_n + nx_p))]), (ref_e + z[int((nx_n + nx_p)):int(na)])])

def surface_n(x):
    return (S_n @ x[int(0):int(ns_n)])

def surface_p(x):
    return (S_p @ x[int(ns_n):int((ns_n + ns_p))])

def dfn_cutoff(t, x, z, nd=None, na=None):
    if nd is None:
        nd = int(x.shape[0])
    if na is None:
        na = int(z.shape[0])
    return (dfn_voltage(t, x, z) - 2.5)

def minimum(q, b):
    result = (q + 0.0)
    for k in range(len(q)):
        if q[int(k)] > b:
            result[int(k)] = b
    return result

def maximum(q, b):
    result = (q + 0.0)
    for k in range(len(q)):
        if q[int(k)] < b:
            result[int(k)] = b
    return result

def softplus(q):
    return (maximum(q, 0.0) + torch.log((1.0 + torch.exp((-torch.abs(q if isinstance(q, torch.Tensor) else torch.tensor(float(q)))) if isinstance((-torch.abs(q if isinstance(q, torch.Tensor) else torch.tensor(float(q)))), torch.Tensor) else torch.tensor(float((-torch.abs(q if isinstance(q, torch.Tensor) else torch.tensor(float(q)))))))) if isinstance((1.0 + torch.exp((-torch.abs(q if isinstance(q, torch.Tensor) else torch.tensor(float(q)))) if isinstance((-torch.abs(q if isinstance(q, torch.Tensor) else torch.tensor(float(q)))), torch.Tensor) else torch.tensor(float((-torch.abs(q if isinstance(q, torch.Tensor) else torch.tensor(float(q)))))))), torch.Tensor) else torch.tensor(float((1.0 + torch.exp((-torch.abs(q if isinstance(q, torch.Tensor) else torch.tensor(float(q)))) if isinstance((-torch.abs(q if isinstance(q, torch.Tensor) else torch.tensor(float(q)))), torch.Tensor) else torch.tensor(float((-torch.abs(q if isinstance(q, torch.Tensor) else torch.tensor(float(q))))))))))))

def sinh(q):
    return ((torch.exp(q if isinstance(q, torch.Tensor) else torch.tensor(float(q))) - torch.exp((-q) if isinstance((-q), torch.Tensor) else torch.tensor(float((-q))))) / 2.0)

def tanh(q):
    result = (q + 0.0)
    for k in range(len(q)):
        if q[int(k)] >= 0.0:
            result[int(k)] = ((1.0 - torch.exp(((-2.0) * q[int(k)]) if isinstance(((-2.0) * q[int(k)]), torch.Tensor) else torch.tensor(float(((-2.0) * q[int(k)]))))) / (1.0 + torch.exp(((-2.0) * q[int(k)]) if isinstance(((-2.0) * q[int(k)]), torch.Tensor) else torch.tensor(float(((-2.0) * q[int(k)]))))))
        else:
            result[int(k)] = ((torch.exp((2.0 * q[int(k)]) if isinstance((2.0 * q[int(k)]), torch.Tensor) else torch.tensor(float((2.0 * q[int(k)])))) - 1.0) / (torch.exp((2.0 * q[int(k)]) if isinstance((2.0 * q[int(k)]), torch.Tensor) else torch.tensor(float((2.0 * q[int(k)])))) + 1.0))
    return result

def ocp_barrier(theta):
    return (205.0568621937484 * (softplus(((-6910.192179565431) * (theta + 0.00077))) - softplus(((-6910.192179565431) * ((1.0 - theta) + 0.00077)))))

def ocp_n(theta):
    q = maximum(minimum(theta, (1.0 - 1e-10)), 1e-10)
    return ((((((1.9793 * torch.exp(((-39.3631) * q) if isinstance(((-39.3631) * q), torch.Tensor) else torch.tensor(float(((-39.3631) * q))))) + 0.2482) - (0.0909 * tanh((29.8538 * (q - 0.1234))))) - (0.04478 * tanh((14.9159 * (q - 0.2769))))) - (0.0205 * tanh((30.4444 * (q - 0.6103))))) + ocp_barrier(theta))

def ocp_p(theta):
    q = maximum(minimum(theta, (1.0 - 1e-10)), 1e-10)
    return (((((((-0.809) * q) + 4.4875) - (0.0428 * tanh((18.5138 * (q - 0.5542))))) - (17.7326 * tanh((15.789 * (q - 0.3117))))) + (17.5842 * tanh((15.9308 * (q - 0.312))))) + ocp_barrier(theta))

def exchange_n(ce, cs):
    q = maximum(minimum(cs, cmax_n), 0.0)
    return ((((6.48e-07 * torch.exp(((35000.0 / gas_constant) * ((1.0 / 298.15) - (1.0 / temperature))) if isinstance(((35000.0 / gas_constant) * ((1.0 / 298.15) - (1.0 / temperature))), torch.Tensor) else torch.tensor(float(((35000.0 / gas_constant) * ((1.0 / 298.15) - (1.0 / temperature))))))) * torch.sqrt(maximum(ce, 0.0) if isinstance(maximum(ce, 0.0), torch.Tensor) else torch.tensor(float(maximum(ce, 0.0))))) * torch.sqrt(q if isinstance(q, torch.Tensor) else torch.tensor(float(q)))) * torch.sqrt((cmax_n - q) if isinstance((cmax_n - q), torch.Tensor) else torch.tensor(float((cmax_n - q)))))

def exchange_p(ce, cs):
    q = maximum(minimum(cs, cmax_p), 0.0)
    return ((((3.42e-06 * torch.exp(((17800.0 / gas_constant) * ((1.0 / 298.15) - (1.0 / temperature))) if isinstance(((17800.0 / gas_constant) * ((1.0 / 298.15) - (1.0 / temperature))), torch.Tensor) else torch.tensor(float(((17800.0 / gas_constant) * ((1.0 / 298.15) - (1.0 / temperature))))))) * torch.sqrt(maximum(ce, 0.0) if isinstance(maximum(ce, 0.0), torch.Tensor) else torch.tensor(float(maximum(ce, 0.0))))) * torch.sqrt(q if isinstance(q, torch.Tensor) else torch.tensor(float(q)))) * torch.sqrt((cmax_p - q) if isinstance((cmax_p - q), torch.Tensor) else torch.tensor(float((cmax_p - q)))))

def electrolyte_diffusivity(ce):
    q = (maximum(ce, 10.0) / 1000.0)
    return (((8.794e-11 * (q ** 2)) - (3.972e-10 * q)) + 4.862e-10)

def electrolyte_conductivity(ce):
    q = (maximum(ce, 10.0) / 1000.0)
    return (((0.1297 * (q ** 3)) - (2.51 * (q ** 1.5))) + (3.329 * q))

def arithmetic_matrix(width, n):
    result = torch.zeros(int((n + 1)), int(n), device=DEVICE)
    result[int(0), int(0)] = 1.5
    result[int(0), int(1)] = (-0.5)
    result[int(n), int((n - 1))] = 1.5
    result[int(n), int((n - 2))] = (-0.5)
    for k in range(int(1), int(n)):
        result[int(k), int((k - 1))] = (width[int(k)] / (width[int((k - 1))] + width[int(k)]))
        result[int(k), int(k)] = (width[int((k - 1))] / (width[int((k - 1))] + width[int(k)]))
    return result

def harmonic_matrix(width, n):
    result = torch.zeros(int((n + 1)), int(n), device=DEVICE)
    result[int(0), int(0)] = 1.0
    result[int(n), int((n - 1))] = 1.0
    for k in range(int(1), int(n)):
        result[int(k), int((k - 1))] = (width[int((k - 1))] / (width[int((k - 1))] + width[int(k)]))
        result[int(k), int(k)] = (width[int(k)] / (width[int((k - 1))] + width[int(k)]))
    return result

def matrix_vector(weights, c, n):
    return torch.stack([torch.sum((weights[int(k)] * c) if isinstance((weights[int(k)] * c), torch.Tensor) else torch.tensor(float((weights[int(k)] * c)))) for _fi_k in range(int(n)) for k in [torch.tensor(float(_fi_k), device=DEVICE)]])

def harmonic_edges(c, weights, n):
    result = (1.0 / matrix_vector(weights, (1.0 / c), (n + 1)))
    result[int(0)] = ((1.5 * c[int(0)]) - (0.5 * c[int(1)]))
    result[int(n)] = ((1.5 * c[int((n - 1))]) - (0.5 * c[int((n - 2))]))
    return result

def particle_matrix(nx, nr, radius, diffusivity):
    size = (nx * nr)
    result = torch.zeros(int(size), int(size), device=DEVICE)
    dr = (radius / nr)
    volume = 0.0
    coeff = 0.0
    for k in range(int(0), int(nx)):
        for q in range(int(0), int(nr)):
            volume = (((((q + 1) ** 3) - (q ** 3)) * (dr ** 3)) / 3.0)
            if q > 0:
                coeff = ((diffusivity * ((q * dr) ** 2)) / (dr * volume))
                result[int(((k * nr) + q)), int((((k * nr) + q) - 1))] = coeff
                result[int(((k * nr) + q)), int(((k * nr) + q))] = (result[int(((k * nr) + q)), int(((k * nr) + q))] - coeff)
            if q < (nr - 1):
                coeff = ((diffusivity * (((q + 1) * dr) ** 2)) / (dr * volume))
                result[int(((k * nr) + q)), int((((k * nr) + q) + 1))] = coeff
                result[int(((k * nr) + q)), int(((k * nr) + q))] = (result[int(((k * nr) + q)), int(((k * nr) + q))] - coeff)
    return result

def particle_source(nx, nr, radius, faraday):
    size = (nx * nr)
    result = torch.zeros(int(size), int(nx), device=DEVICE)
    volume = ((((nr ** 3) - ((nr - 1) ** 3)) * ((radius / nr) ** 3)) / 3.0)
    for k in range(int(0), int(nx)):
        result[int((((k * nr) + nr) - 1)), int(k)] = ((-(radius ** 2)) / (faraday * volume))
    return result

def surface_matrix(nx, nr):
    size = (nx * nr)
    result = torch.zeros(int(nx), int(size), device=DEVICE)
    for k in range(int(0), int(nx)):
        result[int(k), int((((k * nr) + nr) - 1))] = 1.5
        result[int(k), int((((k * nr) + nr) - 2))] = (-0.5)
    return result

def particle_inventory(nx, nr, radius, width, active):
    size = (nx * nr)
    result = torch.zeros(int(size), device=DEVICE)
    for k in range(int(0), int(nx)):
        for q in range(int(0), int(nr)):
            result[int(((k * nr) + q))] = (((active * width) * (((q + 1) ** 3) - (q ** 3))) / (nr ** 3))
    return result

def gradient_values(c, width, n):
    left = c[int(0):int((n - 1))]
    right = c[int(1):int(n)]
    wl = width[int(0):int((n - 1))]
    wr = width[int(1):int(n)]
    internal = ((2.0 * (right - left)) / (wl + wr))
    return torch.cat([torch.tensor([0.0], device=DEVICE), internal, torch.tensor([0.0], device=DEVICE)])

def divergence_values(flux, width, n):
    left = flux[int(0):int(n)]
    right = flux[int(1):int((n + 1))]
    return ((right - left) / width)

# === Program ===
nx_n = 5
nx_s = 5
nx_p = 5
nr_n = 5
nr_p = 5
current_nominal = 5.0
faraday = 96485.33212
gas_constant = 8.31446261815324
temperature = 298.15
length_n = 8.52e-05
length_s = 1.2e-05
length_p = 7.56e-05
area = (0.065 * 1.58)
radius_n = 5.86e-06
radius_p = 5.22e-06
diffusivity_n = 3.3e-14
diffusivity_p = 4e-15
sigma_n = 215.0
sigma_p = 0.18
porosity_n = 0.25
porosity_s = 0.47
porosity_p = 0.335
active_n = 0.75
active_p = 0.665
cmax_n = 33133.0
cmax_p = 63104.0
initial_n = 29866.0
initial_p = 17038.0
initial_e = 1000.0
t_plus = 0.2594
ns_n = (nx_n * nr_n)
ns_p = (nx_p * nr_p)
ne = ((nx_n + nx_s) + nx_p)
nd = (((ns_n + ns_p) + ne) + 1)
na = ((nx_n + nx_p) + ne)
length_total = ((length_n + length_s) + length_p)
a_n = ((3.0 * active_n) / radius_n)
a_p = ((3.0 * active_p) / radius_p)
width_n = torch.stack([((length_n / nx_n) + (0.0 * k)) for _fi_k in range(int(nx_n)) for k in [torch.tensor(float(_fi_k), device=DEVICE)]])
width_s = torch.stack([((length_s / nx_s) + (0.0 * k)) for _fi_k in range(int(nx_s)) for k in [torch.tensor(float(_fi_k), device=DEVICE)]])
width_p = torch.stack([((length_p / nx_p) + (0.0 * k)) for _fi_k in range(int(nx_p)) for k in [torch.tensor(float(_fi_k), device=DEVICE)]])
width = torch.cat([width_n, width_s, width_p])
eps_n = torch.stack([(porosity_n + (0.0 * k)) for _fi_k in range(int(nx_n)) for k in [torch.tensor(float(_fi_k), device=DEVICE)]])
eps_s = torch.stack([(porosity_s + (0.0 * k)) for _fi_k in range(int(nx_s)) for k in [torch.tensor(float(_fi_k), device=DEVICE)]])
eps_p = torch.stack([(porosity_p + (0.0 * k)) for _fi_k in range(int(nx_p)) for k in [torch.tensor(float(_fi_k), device=DEVICE)]])
eps = torch.cat([eps_n, eps_s, eps_p])
transport = (eps ** 1.5)
separator_zero = torch.zeros(int(nx_s), device=DEVICE)
A_e = arithmetic_matrix(width, ne)
H_e = harmonic_matrix(width, ne)
positive_boundary = torch.zeros(int((nx_p + 1)), device=DEVICE)
positive_boundary[int(nx_p)] = 1.0
K_n = particle_matrix(nx_n, nr_n, radius_n, diffusivity_n)
K_p = particle_matrix(nx_p, nr_p, radius_p, diffusivity_p)
B_n = particle_source(nx_n, nr_n, radius_n, faraday)
B_p = particle_source(nx_p, nr_p, radius_p, faraday)
S_n = surface_matrix(nx_n, nr_n)
S_p = surface_matrix(nx_p, nr_p)
cn0 = torch.stack([(initial_n + (0.0 * k)) for _fi_k in range(int(ns_n)) for k in [torch.tensor(float(_fi_k), device=DEVICE)]])
cp0 = torch.stack([(initial_p + (0.0 * k)) for _fi_k in range(int(ns_p)) for k in [torch.tensor(float(_fi_k), device=DEVICE)]])
ce0 = torch.stack([(initial_e + (0.0 * k)) for _fi_k in range(int(ne)) for k in [torch.tensor(float(_fi_k), device=DEVICE)]])
x0 = torch.cat([cn0, cp0, ce0, torch.tensor([0.0], device=DEVICE)])
un0 = ocp_n(torch.stack([torch.as_tensor((initial_n / cmax_n))]))
u_n_initial = un0[int(0)]
up0 = ocp_p(torch.stack([torch.as_tensor((initial_p / cmax_p))]))
u_p_initial = up0[int(0)]
i0n_ref = exchange_n(torch.stack([torch.as_tensor(initial_e)]), torch.stack([torch.as_tensor(initial_n)]))
i0p_ref = exchange_p(torch.stack([torch.as_tensor(initial_e)]), torch.stack([torch.as_tensor(initial_p)]))
eta_n_reference = ((((2.0 * gas_constant) * temperature) / faraday) * asinh_scalar((current_nominal / ((((2.0 * area) * length_n) * a_n) * i0n_ref[int(0)]))))
eta_p_reference = ((((2.0 * gas_constant) * temperature) / faraday) * asinh_scalar(((-current_nominal) / ((((2.0 * area) * length_p) * a_p) * i0p_ref[int(0)]))))
ref_p = (((u_p_initial - u_n_initial) + eta_p_reference) - eta_n_reference)
ref_e = ((-u_n_initial) - eta_n_reference)
z0 = torch.zeros(int(na), device=DEVICE)
sxn = torch.stack([(cmax_n + (0.0 * k)) for _fi_k in range(int(ns_n)) for k in [torch.tensor(float(_fi_k), device=DEVICE)]])
sxp = torch.stack([(cmax_p + (0.0 * k)) for _fi_k in range(int(ns_p)) for k in [torch.tensor(float(_fi_k), device=DEVICE)]])
sxe = torch.stack([(initial_e + (0.0 * k)) for _fi_k in range(int(ne)) for k in [torch.tensor(float(_fi_k), device=DEVICE)]])
x_scale = torch.cat([sxn, sxp, sxe, torch.tensor([5.0], device=DEVICE)])
szn = torch.stack([(0.0001 + (0.0 * k)) for _fi_k in range(int(nx_n)) for k in [torch.tensor(float(_fi_k), device=DEVICE)]])
szp = torch.stack([(0.05 + (0.0 * k)) for _fi_k in range(int(nx_p)) for k in [torch.tensor(float(_fi_k), device=DEVICE)]])
sze = torch.stack([(0.05 + (0.0 * k)) for _fi_k in range(int(ne)) for k in [torch.tensor(float(_fi_k), device=DEVICE)]])
z_scale = torch.cat([szn, szp, sze])
gsn = torch.stack([((((length_total ** 2) * 5.0) / (area * width_n[int(k)])) + (0.0 * k)) for _fi_k in range(int(nx_n)) for k in [torch.tensor(float(_fi_k), device=DEVICE)]])
gsp = torch.stack([((((length_total ** 2) * 5.0) / (area * width_p[int(k)])) + (0.0 * k)) for _fi_k in range(int(nx_p)) for k in [torch.tensor(float(_fi_k), device=DEVICE)]])
gse = torch.stack([((((length_total ** 2) * 5.0) / (area * width[int(k)])) + (0.0 * k)) for _fi_k in range(int(ne)) for k in [torch.tensor(float(_fi_k), device=DEVICE)]])
g_scale = torch.cat([gsn, gsp, gse])
wn = particle_inventory(nx_n, nr_n, radius_n, (length_n / nx_n), active_n)
wp = particle_inventory(nx_p, nr_p, radius_p, (length_p / nx_p), active_p)
inventory_weights = torch.cat([wn, wp, (eps * width), torch.tensor([0.0], device=DEVICE)])