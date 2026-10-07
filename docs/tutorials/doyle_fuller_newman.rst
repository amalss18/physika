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

   python tutorials/battery/run_dfn.py

The command solves the model in float64, compares it with one saved PyBaMM
reference, and writes a trajectory, an error report and four figures to
``tutorials/battery/output/``. It requires no PyBaMM installation.
Settings are explicit in the source: the experiment and mesh are in the
main ``.phyk`` file, parameters are in ``battery/chen2020.phyk``, and time
integration tolerances are in ``battery/dae.py``. There are no tutorial CLI
options.

Read the main Physika file alongside the equations below. Its two imports
contain constitutive laws and conservative spatial operators. Python
supplies stiff time integration, plotting, and access to the external
reference library. The tutorial uses the unchanged Physika core and does not
replace any Physika functions or rewrite the model or its generated code.
The small elementary helpers are defined in ``chen2020.phyk`` with existing
Physika operations. Their branches enforce concentration bounds and stable
evaluation during Newton trials. The runner selects Torch's default float64
dtype before compiling through the existing parser, type checker and AST
code generator. Running the ``.phyk`` file directly defines
the equations; the Python runner performs their time integration.

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
:math:`\dot y=f(t,y,z),\;0=g(t,y,z)`. Physika currently lacks the stiff DAE
integration and matrix linear-solve interfaces needed here, so the small
Python integrator uses Torch autodiff Jacobians and dense Newton solves.
It uses backward Euler for startup, then adaptive variable-step BDF2.
Step doubling estimates concentration error; failed Newton solves or error
tests reduce the step. Algebraic consistency is checked separately.

Initialization solves potentials at fixed initial concentrations with a
Newton line search. A bracketed sequence of implicit solves locates the
2.5 V cutoff. These conditionals implement the selected numerical method.
The integrator contains no alternative methods, switching protocols,
precision modes, backend selection or optional internal-step storage.
Accepted states are detached; autodiff serves Newton's method here.

One reference comparison
------------------------

The saved ``tests/reference/chen2020_discharge_5.npz`` contains a tightly
solved PyBaMM BasicDFN trajectory on the same mesh. Its companion JSON
records parameters, initial concentrations, version, tolerances, potential
reference and regeneration command. The reference is generated once;
normal demonstration runs read it locally.

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

The runner reports separate errors for each field and exits unsuccessfully
if any criterion fails. Tests additionally check finite-volume conservation,
Jacobian accuracy and the integrator's convergence and cutoff location.
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

The fixed demonstration measured 0.044 mV maximum voltage error,
0.052 mV maximum potential error, and 0.121% maximum scaled concentration
error. Cutoff occurred at approximately 3558.196 s, within 0.002 s of the
reference. The checked-in ``demonstration.json`` records the complete run;
wall time depends on the machine.

To regenerate the reference during development, install the PyBaMM
version recorded in its metadata, then run:

.. code-block:: bash

   python tutorials/battery/generate_dfn_reference.py

The model source
----------------

.. literalinclude:: ../../tutorials/doyle_fuller_newman.phyk
   :language: text
