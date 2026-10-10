A forward Doyle–Fuller–Newman battery model
===========================================

This tutorial demonstrates a DFN in Physika: spherical particle diffusion,
electrolyte transport, charge conservation and Butler–Volmer kinetics are
written in ``tutorials/doyle_fuller_newman.phyk``. The example is one
isothermal discharge of an LG M50 cell using Chen2020 parameters:
298.15 K, 5 A, five cells in each spatial coordinate, and a 2.5 V cutoff.
The coarse mesh makes the coupled model small enough to inspect.

Run the demonstration
---------------------

From the repository root, with Physika and its dependencies installed:

.. code-block:: bash

   physika tutorials/doyle_fuller_newman.phyk

The Physika program runs the fixed float32 discharge and prints selected
voltage and discharged-capacity predictions, followed by the cutoff state.
For this experiment the output is approximately:

.. code-block:: text

   time [s]     voltage [V]     discharged capacity [Ah]
      0.000        4.039845              0.000000
    100.000        3.944633              0.138889
   1500.000        3.578698              2.083333
   3000.000        3.228921              4.166667
   3558.189        2.500000              4.941929  (cutoff)

The complete trajectory, reconstructed physical fields and inventory drift
are retained as ``trajectory``, ``prediction_fields`` and ``inventory_drift``
inside the program. It prints only these representative predictions.
Executing the tutorial requires neither PyBaMM nor a battery-specific Python
runner. The full discharge takes several minutes on a CPU.
Settings are explicit in the source: the experiment and mesh are in the
main ``.phyk`` file, parameters are in ``battery/chen2020.phyk``, and time
integration tolerances are in ``battery/dae.phyk``. There are no tutorial CLI
options.

Read the main Physika file alongside the equations below. Its imports
contain constitutive laws, conservative spatial operators, the existing
Gaussian elimination solver and the tutorial's time integrator.
All physics, time integration, physical field reconstruction
and linear solves are written in Physika; this tutorial requires no core
extensions.
The small elementary helpers are defined in ``chen2020.phyk`` with existing
Physika operations. Their branches enforce concentration bounds and stable
evaluation during Newton trials. The final statements call ``integrate_dfn()``,
reconstruct the physical fields, check lithium inventory and display the
selected predictions. The validation figures below are generated during
development and published as static documentation assets; plot-generation
tools are not part of the public tutorial.

Particle diffusion and reaction
-------------------------------

Each electrode contains spherical particles with concentration
:math:`c_{s,k}(r,x,t)`, where :math:`k=n,p` denotes the negative or positive
electrode. Their diffusion equation and boundary conditions are

.. math::

   \frac{\partial c_{s,k}}{\partial t}
   =\frac{1}{r^2}\frac{\partial}{\partial r}
     \left(r^2 D_{s,k}\frac{\partial c_{s,k}}{\partial r}\right),
   \qquad
   \left.\frac{\partial c_{s,k}}{\partial r}\right|_{r=0}=0,
   \qquad
   -D_{s,k}\left.\frac{\partial c_{s,k}}{\partial r}\right|_{r=R_k}
   =\frac{j_k}{F}.

The particle surface concentration and local electrolyte concentration
enter the symmetric Butler–Volmer relation:

.. math::

   j_k=2i_{0,k}(c_e,c_{s,k}^{\mathrm{surf}})
       \sinh\left(\frac{F\eta_k}{2RT}\right),
   \qquad
   \eta_k=\phi_{s,k}-\phi_e-U_k(c_{s,k}^{\mathrm{surf}}/c_{s,k}^{\max}).

``reaction_n`` and ``reaction_p`` express this coupling. The Chen2020
open-circuit potentials and exchange currents are in ``chen2020.phyk``.
Temperature and the Arrhenius factors remain visible even though this
experiment holds temperature fixed. Constitutive guards keep Newton trial
states evaluable; accepted concentration states are not clipped.

Electrolyte transport and potentials
------------------------------------

The electrolyte occupies both electrodes and the separator. Its
conservative molar flux and concentration equation are

.. math::

   N_e=-\epsilon^{1.5}D_e\nabla c_e+\frac{t_+i_e}{F},
   \qquad
   \epsilon\frac{\partial c_e}{\partial t}
      =-\nabla\cdot N_e+\frac{a j}{F}.

The separator has no reaction source. The electrolyte current is

.. math::

   i_e=\epsilon^{1.5}\kappa_e
       \left(\frac{2RT(1-t_+)}{Fc_e}\nabla c_e-\nabla\phi_e\right).

Solid and electrolyte charge conservation determine the potentials:

.. math::

   i_{s,k}=-\sigma_k\nabla\phi_{s,k},
   \qquad \nabla\cdot i_{s,k}+a_kj_k=0,
   \qquad \nabla\cdot i_e-a j=0.

The negative solid current collector is fixed at 0 V. The positive
collector imposes the 5 A current, and its reconstructed potential gives
terminal voltage. Electrolyte outer boundaries have zero flux/current.
``electrolyte_current`` expresses ionic conduction; ``dfn_algebraic``
collects charge balances and ``dfn_rhs`` collects concentration evolution
and discharge capacity.

Spatial discretization and initial states
-----------------------------------------

``finite_volume.phyk`` assembles cell-centred finite volumes using exact
spherical shell volumes for particles. Particle arrays have the radial
coordinate running fastest within each electrode cell. Region interfaces
use cell-width-weighted interpolation. Effective diffusivity and the
thermodynamic factor use harmonic averaging; effective conductivity uses
arithmetic averaging, matching the reference discretization.
Surface concentration and collector potential use linear boundary
extrapolation, :math:`1.5u_N-0.5u_{N-1}` on this uniform mesh.

Initial concentrations are explicit in ``chen2020.phyk``. The main file
assembles them into the differential state
:math:`y=(c_{s,n},c_{s,p},c_e,Q)` and defines the algebraic potential state
:math:`z`. Positive-solid and electrolyte potentials are stored relative
to constant initial equilibrium/kinetic estimates to condition the solve.
``physical_potentials`` reconstructs them before comparison and plotting;
the negative collector still defines the same 0 V reference.

Time integration
----------------

The compiled equations form the semi-explicit DAE
:math:`\dot y=f(t,y,z),\;0=g(t,y,z)`. The tutorial-local ``dae.phyk`` implements
backward Euler for startup, then adaptive variable-step BDF2. It builds
Newton Jacobians with existing scalar ``grad`` calls, one row at a time,
and solves the resulting systems with ``gaussian_solve`` from the existing
:doc:`Gaussian elimination tutorial <linear_solve_gaussian_elimination>`.
That solver uses partial pivoting and is imported with its helper functions.
Step doubling estimates concentration error; failed Newton solves or error
tests reduce the step. Algebraic consistency is checked separately.

Initialization solves potentials at fixed initial concentrations with a
Newton line search. A bracketed sequence of implicit solves locates the
2.5 V cutoff. These conditionals implement the selected numerical method.
The integrator contains no alternative methods, switching protocols,
precision modes, backend selection or optional internal-step storage.
Autodiff serves Newton's method here. Bounded loops and success flags express
iteration with the existing language syntax, and a fixed tensor buffer
stores the trajectory and solver statistics.

The selected float32 tolerances are :math:`10^{-4}` relative and
:math:`10^{-6}` absolute for scaled differential states, :math:`2\times10^{-6}`
for Newton convergence, and :math:`2\times10^{-3}` for scaled algebraic
constraints. The step controller chooses among four scalar factors so that
time arithmetic does not accumulate float32 tensor roundoff. These solver
settings were adjusted for float32; the scientific acceptance limits below
were kept unchanged. Reusing the stricter float64 tolerances caused many
retries and took 913 s in the Python float32 solver.

One reference comparison
------------------------

The saved ``tests/reference/chen2020_discharge_5.npz`` contains a tightly
solved PyBaMM BasicDFN trajectory on the same mesh. Its companion JSON
records parameters, initial concentrations, version, tolerances, potential
reference and generation provenance. The reference is generated during
development; executing the Physika tutorial does not read it.

Voltage, every concentration field and all physical potential fields are
compared at every common saved time and spatial point, spaced 50 s apart.
Particle surface concentrations are also checked. Independently located
cutoff states are assessed through cutoff time rather than treating them
as spatial profiles at the same physical time. Acceptance limits are:

* maximum voltage and per-potential-field error: 1 mV;
* maximum concentration error: 1% of the relevant scale
  (:math:`c_{s,n}^{\max}`, :math:`c_{s,p}^{\max}`, or initial :math:`c_e`);
* relative cutoff-time error: 0.1%;
* relative lithium inventory drift: :math:`10^{-4}`;
* maximum scaled algebraic residual: :math:`2\times10^{-3}`.

The published validation report records separate errors for each field.
Automated tests execute the complete native tutorial once and check the
trajectory against this saved reference without importing PyBaMM.
Fast component tests additionally check finite-volume conservation,
row-wise Jacobian agreement with vectorized autodiff, consistent
initialization and implicit stepping.
This comparison checks implementation agreement on a common coarse mesh;
it does not establish spatial mesh convergence or experimental accuracy.

The figures below compare Physika (solid) with the reference (dashed).
Profile snapshots are at 100, 1500 and 3000 s. Radial profiles use the middle
electrode cell in each electrode; quantitative checks cover all particles.

.. image:: ../_static/dfn_validation/voltage.png
   :alt: Physika and PyBaMM terminal voltage during discharge

.. image:: ../_static/dfn_validation/concentrations.png
   :alt: Electrolyte and particle surface concentration profiles

.. image:: ../_static/dfn_validation/potentials.png
   :alt: Electrolyte and solid potential profiles with a common reference

.. image:: ../_static/dfn_validation/particles.png
   :alt: Radial concentration profiles in representative particles

The float32 Physika demonstration measured 0.119 mV maximum voltage error,
0.119 mV maximum potential error, and 0.537% maximum scaled concentration
error (in the electrolyte). Cutoff occurred at approximately 3558.189 s,
within 0.006 s of the reference. Relative lithium inventory drift was
:math:`5.92\times10^{-7}` and the maximum saved scaled algebraic residual
was :math:`2.97\times10^{-4}`. All acceptance limits passed.

On the same machine with one Torch CPU thread, the complete Physika
integration took 475 s (7.9 minutes). A Python float32 solver with the same
tolerances and step controller took 54 s; both accepted 126 steps with no
rejections. The Physika implementation was 8.7 times slower. The previous
Python float64 demonstration took 72 s with its original stricter settings,
and had 0.044 mV voltage error and 0.121% maximum concentration error.
An earlier Torch-backed linear-solve experiment took 428 s with essentially
the same errors. Reusing the existing Physika Gaussian solver added about
11% to the measured runtime and allowed the core additions to be removed.

Row-by-row autodiff is the main performance limitation: it repeatedly
evaluates residual blocks instead of forming a vectorized Jacobian.
Using ``grad`` on a vector residual currently gives the wrong inferred
shape for this use, so this demonstration uses the existing scalar path
without changing core gradient inference. Float32 meets the demonstration's
accuracy requirements, but moving the solver into Physika currently costs
runtime. The :download:`Gaussian benchmark report <../_static/dfn_validation/gaussian_benchmark.json>`
records the measured runs and settings; wall time depends on the machine.

The reference metadata records the PyBaMM version, Chen2020 parameters,
initial concentrations, spatial mesh and IDAKLU tolerances. The
:download:`validation report <../_static/dfn_validation/demonstration.json>`
and the reference data are retained alongside the published figures.

The model source
----------------

.. literalinclude:: ../../tutorials/doyle_fuller_newman.phyk
   :language: text
