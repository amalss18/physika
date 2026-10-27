Parameter Learning for a Stiff Whole-Cell Host Model
====================================================

In this tutorial we learn two parameters of a **whole cell in miniature**: the
coarse-grained host model of Weiße, Oyarzún, Danos and Swain (2015). Fourteen
ODEs describe how a bacterium imports and metabolizes a nutrient, turns it into
an energy pool, and spends that energy transcribing and translating four classes
of protein: nutrient transporters, metabolic enzymes, housekeeping proteins and
ribosomes. Nothing about growth is imposed. The growth rate *emerges* from how
many ribosomes are busy translating, and every molecule in the cell is diluted
by it. From these mechanisms alone the model reproduces the classic bacterial
growth laws, including Monod's law: growth rises with the external nutrient and
saturates.

Two things make this tutorial different from its systems-biology siblings
(:doc:`/tutorials/repressilator`, :doc:`/tutorials/central_dogma`,
:doc:`/tutorials/chemostat_monod`). First, the model is **stiff**. Its fastest
and slowest timescales differ by a factor of almost four million, so the
explicit RK4 stepper of the other ODE tutorials would need millions of steps
per simulation. We instead write an **implicit backward-Euler integrator** whose
Newton iterations call the differentiable ``gaussian_solve`` from
:doc:`/tutorials/linear_solve_gaussian_elimination`, and we differentiate
straight through every Newton iteration and every linear solve. Second, it is
the largest of the systems-biology models, and it carries the same identifiability lesson
as :doc:`/tutorials/goldbeter_koshland` and :doc:`/tutorials/michaelis_menten`:
growth measured at a single nutrient level cannot separate the two transporter
parameters, while a growth curve across nutrient levels recovers both.


The Model
---------

The state is the number of molecules per cell of fourteen species: internal
nutrient :math:`s_i`, energy :math:`a`, free ribosomes :math:`r`, the proteins
:math:`e_t` (transporter), :math:`e_m` (metabolic enzyme) and :math:`q`
(housekeeping), their mRNAs :math:`m_x` for :math:`x \in \{t, m, q, r\}`, and
the ribosome-mRNA translation complexes :math:`c_x`.

Translation elongation depends on energy, and each complex releases one finished
protein of length :math:`n_x` amino acids at rate :math:`\nu_x`:

.. math::

   \gamma(a) = \frac{\gamma_{\max}\, a}{K_\gamma + a}, \qquad
   \nu_x = \frac{\gamma(a)\, c_x}{n_x}, \qquad
   \lambda = \frac{\gamma(a)}{M} \sum_x c_x .

The last expression is the **growth rate**: the total translation flux in amino
acids per minute divided by the cell mass :math:`M`. Nutrient is imported by the
transporter and converted by the metabolic enzyme, both Michaelis-Menten,

.. math::

   \nu_{\mathrm{imp}} = e_t \,\frac{v_t\, s}{K_t + s}, \qquad
   \nu_{\mathrm{cat}} = e_m \,\frac{v_m\, s_i}{K_m + s_i},

and transcription depends on energy, with the housekeeping genes repressing
themselves:

.. math::

   \omega_r = \frac{w_r\, a}{\theta_r + a}, \qquad
   \omega_t = \omega_m = \frac{w_e\, a}{\theta_{nr} + a}, \qquad
   \omega_q = \frac{w_q\, a}{\theta_{nr} + a} \, \frac{1}{1 + (q/K_q)^{4}} .

The balance equations are then (every species is diluted by growth)

.. math::

   \dot{s}_i &= \nu_{\mathrm{imp}} - \nu_{\mathrm{cat}} - \lambda s_i, \\
   \dot{a}   &= n_s \nu_{\mathrm{cat}} - \sum_x n_x \nu_x - \lambda a, \\
   \dot{e}_x &= \nu_x - \lambda e_x \quad (x = t, m, q), \\
   \dot{r}   &= \nu_r - \lambda r + \sum_x \left( \nu_x - k_b r m_x + k_u c_x \right), \\
   \dot{m}_x &= \omega_x - k_b r m_x + k_u c_x + \nu_x - d_m m_x - \lambda m_x, \\
   \dot{c}_x &= k_b r m_x - k_u c_x - \nu_x - \lambda c_x .

All rates are rational functions, so the model uses only
:math:`+\,-\,*\,/` and powers. The parameter values are the authors' own, taken
from their SBML model (BioModels ``MODEL1502270000``). Note that the paper fits
:math:`K_p = \gamma_{\max}/K_\gamma`, so the program defines
``Kgamma = gmax / Kp``. We learn the transporter's maximal rate and affinity,
:math:`\theta = [v_t, K_t]`, with true values :math:`v_t = 726` and
:math:`K_t = 1000`.

Two sanity checks hold at the steady state, without being imposed anywhere. The
total protein mass :math:`\sum_x n_x p_x + n_r \sum_x c_x` equals :math:`M` to
machine precision (Eq. 10 of the paper), and at :math:`s = 10^4` the cell grows
at :math:`\lambda^\ast = 0.02411` per minute, a doubling time of
28.7 minutes, with ribosomes making up 24.1% of the proteome.


Why the model is stiff
----------------------

At the growing steady state the eigenvalues of the Jacobian span
:math:`8.9\times10^{4}` :math:`\mathrm{min}^{-1}` (the fastest ribosome-mRNA exchange) down
to 0.0241 :math:`\mathrm{min}^{-1}` (dilution by growth): a **stiffness
ratio of** :math:`3.7\times10^{6}`. Explicit RK4 is stable only for
:math:`\rho\,\Delta t \lesssim 2.785`, where :math:`\rho` is the fastest decay rate, i.e. :math:`\Delta t
\lesssim` :math:`3.1\times10^{-5}` minutes, so reaching a steady state that takes many hours of
biological time would require millions of RK4 steps for every simulation, and
every evaluation of the loss needs five of them.

Backward Euler is stable for *any* step size. Each step solves

.. math::

   G(y_{k+1}) = y_{k+1} - y_k - \Delta t\, f(y_{k+1}) = 0

by Newton's method, and each Newton iteration solves the linear system

.. math::

   \left( I - \Delta t\, J \right) w = G, \qquad J = \frac{\partial f}{\partial y},

which is where the linear solver comes in. Because we only need the steady
state, we let :math:`\Delta t` grow geometrically, from :math:`0.05` by a factor
of :math:`1.6` per step, so 30 steps cover the whole approach to steady state.


Step 0: Import the linear solver
--------------------------------

``gaussian_solve`` lives in the Gaussian-elimination tutorial and is
differentiable: it builds its result by concatenation rather than in-place
updates, so gradients flow through the elimination and the back substitution.
We import it together with its helpers, as the Helmholtz tutorial does:

.. code-block:: text

    from tutorials.linear_solve_gaussian_elimination import gaussian_solve, get_2d_array_num_cols, get_2d_array_num_rows, get_1d_array_length, min, max


Step 1: Define the cell
-----------------------

The parameters come first (only :math:`v_t` and :math:`K_t` will be learned),
then the right-hand side ``f``, a direct transcription of the equations above:

.. code-block:: text

    dm: ℝ = 0.1
    kb: ℝ = 1.0
    ku: ℝ = 1.0
    gmax: ℝ = 1260.0
    Kp: ℝ = 180.137803092828
    Kgamma: ℝ = gmax / Kp
    vm: ℝ = 5800.0
    Km: ℝ = 1000.0
    ns: ℝ = 0.5
    nr: ℝ = 7549.0
    nx: ℝ = 300.0
    wr: ℝ = 929.967887456483
    we: ℝ = 4.13917218782445
    wq: ℝ = 948.93498829479
    thetar: ℝ = 426.869333896869
    thetax: ℝ = 4.37973339483464
    Kq: ℝ = 152219.040373749
    M: ℝ = 100000000.0

    def f(y: ℝ[14], vt: ℝ, Kt: ℝ, s: ℝ): ℝ[14]:
        si: ℝ = y[0]
        a: ℝ = y[1]
        r: ℝ = y[2]
        et: ℝ = y[3]
        em: ℝ = y[4]
        q: ℝ = y[5]
        mt: ℝ = y[6]
        mm: ℝ = y[7]
        mq: ℝ = y[8]
        mr: ℝ = y[9]
        ct: ℝ = y[10]
        cm: ℝ = y[11]
        cq: ℝ = y[12]
        cr: ℝ = y[13]
        # translation elongation rate and per-protein translation rates
        gam: ℝ = gmax * a / (Kgamma + a)
        nu_t: ℝ = gam * ct / nx
        nu_m: ℝ = gam * cm / nx
        nu_q: ℝ = gam * cq / nx
        nu_r: ℝ = gam * cr / nr
        # growth rate emerges from the translating ribosomes
        lam: ℝ = gam * (ct + cm + cq + cr) / M
        # nutrient import and metabolism (Michaelis-Menten)
        nu_imp: ℝ = et * vt * s / (Kt + s)
        nu_cat: ℝ = em * vm * si / (Km + si)
        # transcription; housekeeping genes are auto-repressed
        w_r: ℝ = wr * a / (thetar + a)
        w_t: ℝ = we * a / (thetax + a)
        w_m: ℝ = we * a / (thetax + a)
        w_q: ℝ = wq * a / (thetax + a) / (1.0 + (q / Kq) ** 4.0)
        dsi: ℝ = nu_imp - nu_cat - lam * si
        da: ℝ = ns * nu_cat - (nr * nu_r + nx * (nu_t + nu_m + nu_q)) - lam * a
        dr: ℝ = nu_r - lam * r + (nu_r + nu_t + nu_m + nu_q) - kb * r * (mr + mt + mm + mq) + ku * (cr + ct + cm + cq)
        det: ℝ = nu_t - lam * et
        dem: ℝ = nu_m - lam * em
        dq: ℝ = nu_q - lam * q
        dmt: ℝ = w_t - kb * mt * r + ku * ct + nu_t - dm * mt - lam * mt
        dmm: ℝ = w_m - kb * mm * r + ku * cm + nu_m - dm * mm - lam * mm
        dmq: ℝ = w_q - kb * mq * r + ku * cq + nu_q - dm * mq - lam * mq
        dmr: ℝ = w_r - kb * mr * r + ku * cr + nu_r - dm * mr - lam * mr
        dct: ℝ = kb * mt * r - ku * ct - nu_t - lam * ct
        dcm: ℝ = kb * mm * r - ku * cm - nu_m - lam * cm
        dcq: ℝ = kb * mq * r - ku * cq - nu_q - lam * cq
        dcr: ℝ = kb * mr * r - ku * cr - nu_r - lam * cr
        return [dsi, da, dr, det, dem, dq, dmt, dmm, dmq, dmr, dct, dcm, dcq, dcr]

The growth rate is a function of the state alone:

.. code-block:: text

    def growth(y: ℝ[14]): ℝ:
        a: ℝ = y[1]
        return gmax * a / (Kgamma + a) * (y[10] + y[11] + y[12] + y[13]) / M


Step 2: An implicit solver built on ``gaussian_solve``
------------------------------------------------------

The Newton iterations work on :math:`z = \log y`. Every species is a molecule
count, so this keeps the state positive by construction, and Newton cannot jump
to a root with negative molecule counts, which plain Newton on :math:`y` does for
this model when the steps get large. The linear system is the same; only the
update changes, to the damped log-space step :math:`d / (1 + |d|)` with
:math:`d = w / y`. The Jacobian comes from ``grad`` and the solve from
``gaussian_solve``:

.. code-block:: text

    Eye: ℝ[14, 14] = zeros(14, 14)
    for k:ℕ(14):
        Eye[k, k] = 1.0

    def steady(vt: ℝ, Kt: ℝ, s: ℝ, y_init: ℝ[14]): ℝ[14]:
        z: ℝ[14] = log(y_init)
        zk: ℝ[14] = z
        y: ℝ[14] = y_init
        yk: ℝ[14] = y_init
        fk: ℝ[14] = y_init
        w: ℝ[14] = y_init
        d: ℝ[14] = y_init
        J: ℝ[14, 14] = Eye
        A: ℝ[14, 14] = Eye
        dt: ℝ = 0.05
        for k:ℕ(30):
            zk = z
            y = exp(z)
            for j:ℕ(4):
                yk = exp(zk)
                fk = f(yk, vt, Kt, s)
                J = grad(f(yk, vt, Kt, s), yk)
                A = Eye - dt * J
                w = gaussian_solve(A, yk - y - dt * fk)
                d = w / yk
                zk = zk - d / (1.0 + abs(d))
            z = zk
            dt = dt * 1.6
        return exp(z)


Step 3: Generate the measured growth curve
------------------------------------------

The model has two steady states: a dead cell with no energy and no ribosomes,
and a growing one. Starting from a growing cell, we first solve for the steady
state at :math:`s = 10^4` and then use it to warm-start every other solve. The
"measurement" is the steady-state growth rate at five nutrient levels, the
growth curve a microbiologist records:

.. code-block:: text

    y0: ℝ[14] = [500.0, 100.0, 1000.0, 1000.0, 1000.0, 200000.0, 50.0, 50.0, 5000.0, 500.0, 100.0, 100.0, 500.0, 1000.0]
    yss: ℝ[14] = steady(726.0, 1000.0, 10000.0, y0)
    print(growth(yss))

    lt1: ℝ = growth(steady(726.0, 1000.0, 500.0, yss))
    lt2: ℝ = growth(steady(726.0, 1000.0, 1500.0, yss))
    lt3: ℝ = growth(steady(726.0, 1000.0, 5000.0, yss))
    lt4: ℝ = growth(steady(726.0, 1000.0, 20000.0, yss))
    lt5: ℝ = growth(steady(726.0, 1000.0, 100000.0, yss))
    print([lt1, lt2, lt3, lt4, lt5])


Step 4: Loss and gradients through the solver
---------------------------------------------

The loss is the mean squared *relative* error of the five growth rates, and we
optimize the logarithms of the parameters so both stay positive and share a
scale:

.. code-block:: text

    def loss(θ: ℝ[2]): ℝ:
        vt: ℝ = exp(θ[0])
        Kt: ℝ = exp(θ[1])
        e1: ℝ = (growth(steady(vt, Kt, 500.0, yss)) - lt1) / lt1
        e2: ℝ = (growth(steady(vt, Kt, 1500.0, yss)) - lt2) / lt2
        e3: ℝ = (growth(steady(vt, Kt, 5000.0, yss)) - lt3) / lt3
        e4: ℝ = (growth(steady(vt, Kt, 20000.0, yss)) - lt4) / lt4
        e5: ℝ = (growth(steady(vt, Kt, 100000.0, yss)) - lt5) / lt5
        return (e1 * e1 + e2 * e2 + e3 * e3 + e4 * e4 + e5 * e5) / 5.0

There is no hand-written adjoint here. ``grad(loss, θ)`` differentiates through
the whole computation: 5 solves, 30 backward-Euler steps each, 4 Newton
iterations per step, each with a Jacobian and a Gaussian elimination. We
checked this gradient against the exact implicit-function derivative of the
steady state, :math:`d y^\ast/d\theta = -J^{-1}\, \partial f/\partial \theta`,
in a double-precision reference implementation; the two agree to
10 to 11 significant digits. (``grad`` treats the Jacobian
inside each Newton iteration as a constant. That is harmless here: the term it
leaves out is multiplied by the Newton residual, which vanishes at convergence.)


Step 5: Train with Adam
-----------------------

We start from a deliberately wrong guess, :math:`v_t = 400` and
:math:`K_t = 3000`, and run Adam on the log-parameters:

.. code-block:: text

    θ: ℝ[2] = [log(400.0), log(3000.0)]
    print(loss(θ))
    learning_rate: ℝ = 0.05
    beta1: ℝ = 0.9
    beta2: ℝ = 0.999
    eps_adam: ℝ = 0.00000001
    m_adam: ℝ[2] = [0.0, 0.0]
    v_adam: ℝ[2] = [0.0, 0.0]
    t_adam: ℝ = 0.0

    # ------------------------------------------------------------------
    # set epochs higher (e.g. 200) for real training; here it is 1 so
    # physika's test suite (tests/test_tutorials.py) runs quickly
    # ------------------------------------------------------------------
    epochs: ℕ = 1

    for ep:ℕ(epochs):
        g = grad(loss, θ)
        t_adam = t_adam + 1.0
        m_adam = beta1 * m_adam + (1.0 - beta1) * g
        v_adam = beta2 * v_adam + (1.0 - beta2) * (g * g)
        mhat = m_adam / (1.0 - beta1 ** t_adam)
        vhat = v_adam / (1.0 - beta2 ** t_adam)
        θ = θ - learning_rate * mhat / (sqrt(vhat) + eps_adam)
        print(θ)

    print(loss(θ))

.. note::
   The committed ``tutorials/weisse_host_model.phyk`` sets ``epochs = 1`` so
   the test suite runs quickly (one epoch takes about 10 seconds). Raise it
   to ``200`` to fit the model.


Step 6: Results
---------------

After 200 Adam epochs the loss falls from :math:`2.8\times10^{-1}` to
:math:`1.9\times10^{-10}` as Physika computes it in single precision
(:math:`1.2\times10^{-10}` re-evaluated in double precision), more than nine
orders of magnitude, and both
transporter parameters are recovered: :math:`v_t = 725.98` and
:math:`K_t = 999.98` against the true :math:`726` and :math:`1000` (errors of
0.002% and 0.003%), from a start that was 45% and 200% off. A double-precision
PyTorch implementation of the same solver, run from the same start, lands on
:math:`v_t = 725.98`, :math:`K_t = 999.97`, so Physika's single-precision
arithmetic does not limit the fit. The whole run takes about 32 minutes on a
CPU.

.. figure:: /_static/tutorial_files/output_weisse_host_model.png
   :alt: Growth curve fit, training loss and Jacobian spectrum of the Weisse host model
   :align: center
   :width: 800px

   **A.** Steady-state growth rate against external nutrient (Monod's law). The
   learned growth curve lies on the truth across the whole range, including
   nutrient levels between the five measured points; the initial guess grows
   far too slowly. **B.** Training loss along the Physika run (re-evaluated in
   double precision from the logged parameters). **C.** Decay rates of the
   Jacobian at the growing steady state: they span from dilution by growth to
   ribosome-mRNA exchange, a stiffness ratio of about :math:`3.7\times10^{6}`,
   which is why the tutorial integrates implicitly.

The instructive failure is to measure growth at **one** nutrient level. The
external nutrient enters the model only through the import term
:math:`e_t\, v_t s/(K_t + s)`, so at a fixed :math:`s` the parameters act only
through the single combination :math:`v_t/(K_t + s)`. Every pair on that curve
produces exactly the same cell. For example, at :math:`s = 500` the pairs
:math:`(1210, 2000)` and :math:`(2178, 4000)` give the same growth rate as the
truth to ten digits, yet they overpredict growth at :math:`s = 10^5` by 37% and
83%. Measurements at two or more nutrient levels, ideally on both sides of
:math:`K_t`, separate the transporter's capacity from its affinity, which is the same lesson as the single-dose fit of
:doc:`/tutorials/goldbeter_koshland` and the low-substrate fit of
:doc:`/tutorials/michaelis_menten`.


Full Code
---------

.. code-block:: text

    # ============================================================
    # Parameter Learning for a Stiff Whole-Cell Host Model
    # (Weisse, Oyarzun, Danos & Swain, PNAS 2015)
    # ============================================================
    # A coarse-grained "cell in miniature": 14 ODEs for nutrient
    # import and metabolism, an energy pool, and the transcription,
    # ribosome binding and translation of four protein classes
    # (transporters et, metabolic enzymes em, housekeeping q, and
    # ribosomes r). Nothing about growth is imposed: the growth rate
    #
    #     lambda = gamma(a) * (ct + cm + cq + cr) / M
    #
    # EMERGES from how many ribosomes are busy translating, and every
    # species is diluted by it. The model reproduces the bacterial
    # growth laws, including Monod's law: lambda rises with the
    # external nutrient s and saturates.
    #
    # THE TWIST IS STIFFNESS. At the living steady state the fastest
    # Jacobian mode is ~9e4 /min while the slowest (dilution by growth)
    # is ~0.024 /min: a stiffness ratio of ~3.7e6. Explicit RK4 would
    # need dt < ~3e-5 min, i.e. millions of steps per simulation. So we use an IMPLICIT backward-Euler
    # integrator: every step solves G(y') = y' - y - dt*f(y') = 0 by
    # Newton's method, and every Newton iteration solves the linear
    # system (I - dt*J) * w = G with Physika's own differentiable
    # gaussian_solve, where J = grad(f, y) is the Jacobian. Because
    # backward Euler is stable for any dt, we grow dt geometrically
    # and march straight to the steady state.
    #
    # We learn theta = [vt, Kt], the transporter's maximal import rate
    # and its affinity for nutrient, from a measured growth curve: the
    # steady-state growth rate at five nutrient levels. Gradients flow
    # through every Newton iteration and every linear solve.
    # ============================================================

    # importing gaussian_solve function from `tutorials/`, in future gaussian_solve will
    # get imported from stdlib
    from tutorials.linear_solve_gaussian_elimination import gaussian_solve, get_2d_array_num_cols, get_2d_array_num_rows, get_1d_array_length, min, max

    # ---- model parameters (Weisse et al. 2015, values from the authors' SBML
    # model, BioModels MODEL1502270000); vt and Kt are learned ----
    dm: ℝ = 0.1
    kb: ℝ = 1.0
    ku: ℝ = 1.0
    gmax: ℝ = 1260.0
    Kp: ℝ = 180.137803092828
    Kgamma: ℝ = gmax / Kp
    vm: ℝ = 5800.0
    Km: ℝ = 1000.0
    ns: ℝ = 0.5
    nr: ℝ = 7549.0
    nx: ℝ = 300.0
    wr: ℝ = 929.967887456483
    we: ℝ = 4.13917218782445
    wq: ℝ = 948.93498829479
    thetar: ℝ = 426.869333896869
    thetax: ℝ = 4.37973339483464
    Kq: ℝ = 152219.040373749
    M: ℝ = 100000000.0

    # state y = [si, a, r, et, em, q, mt, mm, mq, mr, ct, cm, cq, cr]
    def f(y: ℝ[14], vt: ℝ, Kt: ℝ, s: ℝ): ℝ[14]:
        si: ℝ = y[0]
        a: ℝ = y[1]
        r: ℝ = y[2]
        et: ℝ = y[3]
        em: ℝ = y[4]
        q: ℝ = y[5]
        mt: ℝ = y[6]
        mm: ℝ = y[7]
        mq: ℝ = y[8]
        mr: ℝ = y[9]
        ct: ℝ = y[10]
        cm: ℝ = y[11]
        cq: ℝ = y[12]
        cr: ℝ = y[13]
        # translation elongation rate and per-protein translation rates
        gam: ℝ = gmax * a / (Kgamma + a)
        nu_t: ℝ = gam * ct / nx
        nu_m: ℝ = gam * cm / nx
        nu_q: ℝ = gam * cq / nx
        nu_r: ℝ = gam * cr / nr
        # growth rate emerges from the translating ribosomes
        lam: ℝ = gam * (ct + cm + cq + cr) / M
        # nutrient import and metabolism (Michaelis-Menten)
        nu_imp: ℝ = et * vt * s / (Kt + s)
        nu_cat: ℝ = em * vm * si / (Km + si)
        # transcription; housekeeping genes are auto-repressed
        w_r: ℝ = wr * a / (thetar + a)
        w_t: ℝ = we * a / (thetax + a)
        w_m: ℝ = we * a / (thetax + a)
        w_q: ℝ = wq * a / (thetax + a) / (1.0 + (q / Kq) ** 4.0)
        dsi: ℝ = nu_imp - nu_cat - lam * si
        da: ℝ = ns * nu_cat - (nr * nu_r + nx * (nu_t + nu_m + nu_q)) - lam * a
        dr: ℝ = nu_r - lam * r + (nu_r + nu_t + nu_m + nu_q) - kb * r * (mr + mt + mm + mq) + ku * (cr + ct + cm + cq)
        det: ℝ = nu_t - lam * et
        dem: ℝ = nu_m - lam * em
        dq: ℝ = nu_q - lam * q
        dmt: ℝ = w_t - kb * mt * r + ku * ct + nu_t - dm * mt - lam * mt
        dmm: ℝ = w_m - kb * mm * r + ku * cm + nu_m - dm * mm - lam * mm
        dmq: ℝ = w_q - kb * mq * r + ku * cq + nu_q - dm * mq - lam * mq
        dmr: ℝ = w_r - kb * mr * r + ku * cr + nu_r - dm * mr - lam * mr
        dct: ℝ = kb * mt * r - ku * ct - nu_t - lam * ct
        dcm: ℝ = kb * mm * r - ku * cm - nu_m - lam * cm
        dcq: ℝ = kb * mq * r - ku * cq - nu_q - lam * cq
        dcr: ℝ = kb * mr * r - ku * cr - nu_r - lam * cr
        return [dsi, da, dr, det, dem, dq, dmt, dmm, dmq, dmr, dct, dcm, dcq, dcr]

    def growth(y: ℝ[14]): ℝ:
        a: ℝ = y[1]
        return gmax * a / (Kgamma + a) * (y[10] + y[11] + y[12] + y[13]) / M

    Eye: ℝ[14, 14] = zeros(14, 14)
    for k:ℕ(14):
        Eye[k, k] = 1.0

    # Backward Euler with a geometrically growing step, 4 Newton
    # iterations per step. Every species is a molecule count, so we
    # iterate on z = log(y): the state can never turn negative and
    # Newton cannot jump to a non-physical root. Each Newton iteration
    # solves the linear system
    #     (Eye - dt * J) * w = yk - y - dt * f(yk)
    # with gaussian_solve, J being the Jacobian of f at yk, and then
    # takes the damped log-space step d / (1 + |d|) with d = w / yk.
    def steady(vt: ℝ, Kt: ℝ, s: ℝ, y_init: ℝ[14]): ℝ[14]:
        z: ℝ[14] = log(y_init)
        zk: ℝ[14] = z
        y: ℝ[14] = y_init
        yk: ℝ[14] = y_init
        fk: ℝ[14] = y_init
        w: ℝ[14] = y_init
        d: ℝ[14] = y_init
        J: ℝ[14, 14] = Eye
        A: ℝ[14, 14] = Eye
        dt: ℝ = 0.05
        for k:ℕ(30):
            zk = z
            y = exp(z)
            for j:ℕ(4):
                yk = exp(zk)
                fk = f(yk, vt, Kt, s)
                J = grad(f(yk, vt, Kt, s), yk)
                A = Eye - dt * J
                w = gaussian_solve(A, yk - y - dt * fk)
                d = w / yk
                zk = zk - d / (1.0 + abs(d))
            z = zk
            dt = dt * 1.6
        return exp(z)

    # The model has two steady states: a dead cell (no energy, no
    # ribosomes) and a growing one. We start from a growing cell; its
    # steady state at s = 1e4 then warm-starts every other solve.
    y0: ℝ[14] = [500.0, 100.0, 1000.0, 1000.0, 1000.0, 200000.0, 50.0, 50.0, 5000.0, 500.0, 100.0, 100.0, 500.0, 1000.0]
    yss: ℝ[14] = steady(726.0, 1000.0, 10000.0, y0)
    print(growth(yss))

    # "measured" growth curve lambda(s) at the true [vt, Kt] = [726, 1000]
    lt1: ℝ = growth(steady(726.0, 1000.0, 500.0, yss))
    lt2: ℝ = growth(steady(726.0, 1000.0, 1500.0, yss))
    lt3: ℝ = growth(steady(726.0, 1000.0, 5000.0, yss))
    lt4: ℝ = growth(steady(726.0, 1000.0, 20000.0, yss))
    lt5: ℝ = growth(steady(726.0, 1000.0, 100000.0, yss))
    print([lt1, lt2, lt3, lt4, lt5])

    # relative mean squared error over the five nutrient levels;
    # theta = [log vt, log Kt] keeps both positive and on a common scale
    def loss(θ: ℝ[2]): ℝ:
        vt: ℝ = exp(θ[0])
        Kt: ℝ = exp(θ[1])
        e1: ℝ = (growth(steady(vt, Kt, 500.0, yss)) - lt1) / lt1
        e2: ℝ = (growth(steady(vt, Kt, 1500.0, yss)) - lt2) / lt2
        e3: ℝ = (growth(steady(vt, Kt, 5000.0, yss)) - lt3) / lt3
        e4: ℝ = (growth(steady(vt, Kt, 20000.0, yss)) - lt4) / lt4
        e5: ℝ = (growth(steady(vt, Kt, 100000.0, yss)) - lt5) / lt5
        return (e1 * e1 + e2 * e2 + e3 * e3 + e4 * e4 + e5 * e5) / 5.0

    # deliberately wrong start: vt = 400, Kt = 3000
    θ: ℝ[2] = [log(400.0), log(3000.0)]
    print(loss(θ))
    learning_rate: ℝ = 0.05
    beta1: ℝ = 0.9
    beta2: ℝ = 0.999
    eps_adam: ℝ = 0.00000001
    m_adam: ℝ[2] = [0.0, 0.0]
    v_adam: ℝ[2] = [0.0, 0.0]
    t_adam: ℝ = 0.0

    # ------------------------------------------------------------------
    # set epochs higher (e.g. 200) for real training; here it is 1 so
    # physika's test suite (tests/test_tutorials.py) runs quickly
    # ------------------------------------------------------------------
    epochs: ℕ = 1

    for ep:ℕ(epochs):
        g = grad(loss, θ)
        t_adam = t_adam + 1.0
        m_adam = beta1 * m_adam + (1.0 - beta1) * g
        v_adam = beta2 * v_adam + (1.0 - beta2) * (g * g)
        mhat = m_adam / (1.0 - beta1 ** t_adam)
        vhat = v_adam / (1.0 - beta2 ** t_adam)
        θ = θ - learning_rate * mhat / (sqrt(vhat) + eps_adam)
        print(θ)

    print(loss(θ))


References
----------

- A. Y. Weiße, D. A. Oyarzún, V. Danos and P. S. Swain, *Mechanistic links between cellular trade-offs, gene expression, and growth*, Proc. Natl. Acad. Sci. USA 112(9), E1038-E1047 (2015).
- BioModels Database, model ``MODEL1502270000`` (Weiße et al. 2015, SBML submitted by the authors).
- M. Scott, C. W. Gunderson, E. M. Mateescu, Z. Zhang and T. Hwa, *Interdependence of cell growth and gene expression: origins and consequences*, Science 330(6007), 1099-1102 (2010).
- J. Monod, *The growth of bacterial cultures*, Annual Review of Microbiology 3, 371-394 (1949).
- E. Hairer and G. Wanner, *Solving Ordinary Differential Equations II: Stiff and Differential-Algebraic Problems*, 2nd ed. (Springer, 1996).
