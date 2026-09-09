"""TORAX interface for the EPEDNN-mit pedestal models.

Two machines are supported, selected via the ``machine`` config field:

- ``"sparc"``: valid for the SPARC parameter space, as specified in
  https://github.com/aaronkho/epednn_mit/tree/main/src/epednn_mit/models/sparc.
  Please cite [M. Muraca et al. 2025 Nucl. Fusion 65
  096010](https://doi.org/10.1088/1741-4326/adf656) in any works using it.
- ``"scoping"``: trained for high-field FPP scoping studies on dimensionless
  inputs, as specified in
  https://github.com/aaronkho/epednn_mit/tree/scoping_model/src/epednn_mit/models/scoping.
  A citation for this model is pending.

This file defines the four necessary classes to use the model in TORAX:
1. A network wrapper, which closes over the EPEDNN-mit model, parameters, and
   statistics. This is necessary to make the model hashable. Each network is a
   subclass of ``BaseEPEDNNmitModel`` declaring its weights and its named
   inputs; everything else -- loading, input packing, clipping -- is shared.
2. A TORAX pedestal model, which derives the network-independent quantities
   from TORAX state and calls the wrapper within its _call_implementation.
   This is what will be called by TORAX.
3. A TORAX pedestal model Pydantic config, which is used to build the pedestal
   model and runtime params. This allows us to use the model from a TORAX
   config.
4. A runtime params class, which contains any input parameters specific to the
   EPEDNN-mit model. This will be passed to the pedestal model by TORAX at each
   time step.
"""

import dataclasses
import pathlib
import types
from typing import Annotated, Any, ClassVar, Final, Literal, TypeAlias

import chex
from epednn_mit.models.scoping import jax_model as epednn_mit_scoping_jax_model
from epednn_mit.models.sparc import jax_model as epednn_mit_sparc_jax_model
import jax
from jax import numpy as jnp
import jaxtyping as jt
import torax
from torax._src import constants
from torax._src.pedestal_model import pedestal_model as pedestal_model_lib
from torax._src.pedestal_model import pedestal_model_output as pedestal_model_output_lib
from torax._src.pedestal_model import pedestal_transition_state as pedestal_transition_state_lib
from torax._src.physics import formulas
from torax._src.torax_pydantic import torax_pydantic
from typing_extensions import override

# pylint: disable=invalid-name


EPEDNNmitStats: TypeAlias = dict[str, jax.Array]
EPEDNNmitParams: TypeAlias = dict[str, Any]
EPEDNNmitMachine: TypeAlias = Literal["sparc", "scoping"]

# Each network's inputs, in the order the network expects them: the network's
# own input name (per its README) mapped to the quantity it is drawn from and
# its training range. Iterating one dict drives both the input vector and its
# clip bounds, so the two cannot fall out of order.
#
# Source keys refer either to `_derive_torax_quantities` output or to a
# network's own `constants`. Names are qualified where the two networks draw
# the same physical quantity from different places -- `kappa`/`delta` at the
# LCFS for sparc versus psi_N = 0.995 for scoping, and a frozen device minor
# radius for sparc versus the geometry's for scoping.
InputSpecs: TypeAlias = dict[str, tuple[str, float, float]]

# For definitions of these parameters, see the EPEDNN-mit SPARC README:
# https://github.com/aaronkho/epednn_mit/blob/main/src/epednn_mit/models/sparc/README.txt
_SPARC_INPUT_SPECS: Final[InputSpecs] = {
    "Ip": ("Ip_MA", 1.6, 14.3),
    "Bt": ("B_0", 7.2, 12.2),
    "R": ("R_device", 1.85, 1.85),
    "a": ("a_device", 0.57, 0.57),
    "kappa": ("kappa_lcfs", 1.53, 2.29),
    "delta": ("delta_lcfs", 0.39, 0.59),
    "neped": ("n_e_ped_1e19", 2.84, 90.235),
    "betan": ("beta_N", 0.8, 1.6),
    "zeff": ("Z_eff_ped", 1.3, 2.5),
}
_SPARC_DEVICE_MAJOR_RADIUS: Final[float] = 1.85
_SPARC_DEVICE_MINOR_RADIUS: Final[float] = 0.57

# For definitions of these parameters, see the EPEDNN-mit scoping README:
# https://github.com/aaronkho/epednn_mit/blob/scoping_model/src/epednn_mit/models/scoping/README.txt
_SCOPING_INPUT_SPECS: Final[InputSpecs] = {
    "a": ("a_minor", 0.4, 2.2),
    "aspect": ("aspect", 2.0, 4.2),
    "kappa": ("kappa_995", 1.3, 2.5),
    "delta": ("delta_995", 0.3, 0.7),
    # README: "not a clean boundary so (3.0, 17.0) might be more prudent".
    "bt": ("B_0", 2.0, 18.0),
    "qstar": ("qstar", 3.0, 5.0),
    "betan": ("beta_N", 0.3, 3.7),
    "zeff": ("Z_eff_ped", 1.2, 3.2),
    "fgped": ("fgped", 0.3, 1.3),
    "nsfrac": ("nsfrac", 0.2, 0.8),
    "tesep": ("tesep", 50.0, 500.0),  # [eV]
}

# psi_N surface where the scoping shape parameters (kappa, delta) are sampled.
# EPED internals do not model X-points, so LCFS shaping values that include
# them can throw off the prediction; psi_N = 0.995 is the recommended
# approximation when coupling to a core transport model (used by MAESTRO;
# A. Ho, private communication, July 2026).
_SCOPING_SHAPE_PSI_NORM: Final[float] = 0.995


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams(torax.pedestal.RuntimeParams):
  """Runtime params for the EPEDNNmitPedestalModel."""

  n_e_ped: jt.Float[jt.Scalar, ""]  # [m^-3]
  T_i_T_e_ratio: jt.Float[jt.Scalar, ""]
  P_ped_multiplier: jt.Float[jt.Scalar, ""]
  # Scoping-model inputs not derivable from TORAX state (unused by "sparc").
  nsfrac: jt.Float[jt.Scalar, ""]  # n_e_sep / n_e_ped
  tesep: jt.Float[jt.Scalar, ""]  # [eV]


class BaseEPEDNNmitModel:
  """One EPEDNN-mit network, with its weights and named inputs.

  Captures the EPEDNN-mit model, parameters, and statistics as attributes of
  the class. Hashes by value (machine name), not id.

  Everything that runs -- weight loading, input selection, clipping,
  inference -- is shared. Only data differs between networks, so a subclass
  is a declaration: set ``machine``, ``jax_model``, ``weights_glob``,
  ``input_specs``, and ``constants`` if the network takes any fixed inputs.
  """

  machine: ClassVar[EPEDNNmitMachine]
  # The sparc and scoping jax modules share an API but differ in architecture
  # and pickle key map, so each network loads through its own module.
  jax_model: ClassVar[types.ModuleType]
  weights_glob: ClassVar[str]
  # The network's inputs, in the order the network expects them.
  input_specs: ClassVar[InputSpecs]
  # Network inputs that are fixed rather than derived from TORAX state.
  constants: ClassVar[dict[str, float]] = {}

  def __init__(self):
    model_dir = pathlib.Path(self.jax_model.__file__).parent
    model_weights = sorted(model_dir.glob(self.weights_glob))
    self._stats, self._params = (
        self.jax_model.load_ensemble_params_from_pickle(model_weights)
    )
    self.model = self.jax_model.EPEDNNmitEnsemble()

  @property
  def input_lower_bounds(self) -> jax.Array:
    return jnp.array([lo for _, lo, _ in self.input_specs.values()])

  @property
  def input_upper_bounds(self) -> jax.Array:
    return jnp.array([hi for _, _, hi in self.input_specs.values()])

  def pack_inputs(self, quantities: dict[str, Any]) -> jax.Array:
    """Selects, orders and clips this network's inputs.

    One pass over ``input_specs`` drives the value lookup and both clip
    bounds, so an input can never be clipped against another's range.

    Args:
      quantities: Named quantities derived from TORAX state, as returned by
        ``EPEDNNmitPedestalModel._derive_torax_quantities``.
    """
    available = quantities | self.constants
    raw_inputs = jnp.array(
        # Networks were trained with float32. The clip promotes back to the
        # bounds' dtype, which is what the network must see: evaluated at a
        # precision its loaded weights and normalisation statistics are not
        # in, it returns a value ~4e6 too large instead of failing.
        [available[source] for source, _, _ in self.input_specs.values()],
        dtype=jnp.float32,
    )
    return jnp.clip(
        raw_inputs, self.input_lower_bounds, self.input_upper_bounds
    )

  def out_of_range_inputs(
      self, quantities: dict[str, Any]
  ) -> dict[str, tuple[float, float, float]]:
    """Maps each input outside its training range to ``(value, min, max)``.

    ``pack_inputs`` clips silently, so a prediction can look reasonable while
    resting on a clipped input. For diagnostics, off the traced path.
    """
    available = quantities | self.constants
    return {
        name: (float(available[source]), lo, hi)
        for name, (source, lo, hi) in self.input_specs.items()
        if not lo <= float(available[source]) <= hi
    }

  def __call__(self, inputs: jax.Array) -> tuple[jax.Array, jax.Array]:
    P_ped_kPa, pedestal_width_psi_norm = self.model.apply(
        self._params, inputs, **self._stats
    )
    return P_ped_kPa, pedestal_width_psi_norm

  def __hash__(self):
    return hash(self.machine)


class SparcModel(BaseEPEDNNmitModel):
  """EPEDNN-mit trained on the SPARC parameter space."""

  machine = "sparc"
  jax_model = epednn_mit_sparc_jax_model
  weights_glob = "epednn_mit_sparc_*.pkl"
  input_specs = _SPARC_INPUT_SPECS
  # SPARC's geometry is baked into the training set, so the network takes the
  # device major and minor radius as fixed inputs.
  constants = {
      "R_device": _SPARC_DEVICE_MAJOR_RADIUS,
      "a_device": _SPARC_DEVICE_MINOR_RADIUS,
  }


class ScopingModel(BaseEPEDNNmitModel):
  """EPEDNN-mit trained on dimensionless high-field FPP scoping inputs.

  Every input is drawn from TORAX state or config, so there are no fixed
  device constants.
  """

  machine = "scoping"
  jax_model = epednn_mit_scoping_jax_model
  weights_glob = "epednn_mit_scoping_*.pkl"
  input_specs = _SCOPING_INPUT_SPECS


MODELS: Final[dict[str, type[BaseEPEDNNmitModel]]] = {
    model.machine: model for model in (SparcModel, ScopingModel)
}


@dataclasses.dataclass(frozen=True, eq=False)
class EPEDNNmitPedestalModel(pedestal_model_lib.PedestalModel):
  """TORAX pedestal model using EPEDNN-mit to predict pressure and width."""

  machine: EPEDNNmitMachine = "sparc"
  # The following fields are set by __post_init__.
  model: BaseEPEDNNmitModel = dataclasses.field(init=False)

  def __post_init__(self):
    if self.machine not in MODELS:
      raise ValueError(
          f"Unsupported machine: {self.machine}. Supported machines are "
          f"{sorted(MODELS)}."
      )
    # Need to use __setattr__  to install attributes as this is a frozen
    # dataclass.
    object.__setattr__(self, "model", MODELS[self.machine]())
    super().__post_init__()

  def _derive_torax_quantities(
      self,
      runtime_params: torax.RuntimeParams,
      geo: torax.Geometry,
      core_profiles: torax.CoreProfiles,
      previous_rho_norm_ped_top: jax.Array | None = None,
  ) -> dict[str, Any]:
    """Derives every quantity any EPEDNN-mit network takes as an input.

    Network-independent: the selected network picks what it needs by name in
    ``BaseEPEDNNmitModel.pack_inputs``. Quantities no network reads are
    dropped by XLA, so deriving all of them costs nothing.

    When ``previous_rho_norm_ped_top`` is provided (and is not the
    placeholder ``inf``), Z_eff and n_e_ped are evaluated at the mtanh
    "pedestal" location (ψ_ped = 1 - Δ). n_e_top is read from the
    evolved ``core_profiles.n_e`` at rho_norm_top, then corrected to ψ_ped
    using the assumed mtanh relationship below.

    mtanh pedestal density profile (Snyder et al., PPCF 46, 2004):

    Neglecting the contribution of the core density profile:

      n(ψ) = n_sep + a₀·[tanh(1) - tanh(2(ψ - ψ_mid)/Δ)]

    where ψ_mid = 1 - Δ/2.  Key locations:

      ψ_top = 1 - 1.5Δ  →  tanh arg = -2  →  EPED-NN output / TORAX top
      ψ_ped = 1 - Δ      →  tanh arg = -1  →  EPED-NN input location
      ψ_mid = 1 - Δ/2    →  tanh arg =  0  →  inflection point

    Profile values at these locations::

      n_top = n_sep + a₀·(tanh 1 + tanh 2)
      n_ped = n_sep + a₀·2·tanh 1

    Eliminating a₀:

      n_ped = n_sep + (n_top - n_sep)·C
      C = 2·tanh(1) / (tanh(1) + tanh(2)) ≈ 0.883

    Args:
      runtime_params: Runtime parameters.
      geo: Geometry.
      core_profiles: Core plasma profiles.
      previous_rho_norm_ped_top: Previous timestep's rho_norm at the pedestal
        top. ``jnp.inf`` signals first timestep (placeholder).

    Returns:
      Named quantities, keyed by the source names in the networks'
      ``input_specs``.
    """
    assert isinstance(runtime_params.pedestal, RuntimeParams)

    _, _, beta_N = formulas.calculate_betas(core_profiles, geo)

    # -- Convert ρ_top (pedestal top) to ρ_ped (mtanh pedestal location) --
    # EPED-NN expects inputs evaluated at ψ_ped = 1 − Δ, but TORAX tracks
    # ρ_top (= ψ_top = 1 − 1.5Δ). This section maps ρ_top → ψ_top → Δ →
    # ψ_ped → ρ_ped so we can sample profiles at the correct location.

    # Default ρ_top to 0.9 when no previous pedestal output is available.
    if previous_rho_norm_ped_top is not None:
      safe_rho_top = jnp.where(
          jnp.isinf(previous_rho_norm_ped_top),
          jnp.float32(0.9),
          previous_rho_norm_ped_top,
      )
    else:
      safe_rho_top = jnp.float32(0.9)

    # Map ρ → ψ using the normalised poloidal flux profile.
    psi_face = core_profiles.psi.face_value()
    psi_norm = (core_profiles.psi.value - psi_face[0]) / (
        psi_face[-1] - psi_face[0]
    )
    psi_top = jnp.interp(safe_rho_top, geo.rho_norm, psi_norm)

    # Δ = (1 − ψ_top) / 1.5  ;  ψ_ped = 1 − Δ
    delta_psi = (1.0 - psi_top) / 1.5
    psi_ped = 1.0 - delta_psi

    # Map ψ_ped back to ρ_ped.
    rho_ped = jnp.interp(psi_ped, psi_norm, geo.rho_norm)

    # -- n_e from profile at ρ_top, then mtanh correction to ψ_ped --
    n_e_top = jnp.interp(safe_rho_top, geo.rho_norm, core_profiles.n_e.value)
    _C = 2.0 * jnp.tanh(1.0) / (jnp.tanh(1.0) + jnp.tanh(2.0))
    n_e_sep = core_profiles.n_e.face_value()[-1]
    n_e_ped = n_e_sep + (n_e_top - n_e_sep) * _C

    # -- Z_eff at the pedestal location --
    ped_idx = jnp.argmin(jnp.abs(geo.rho_norm - rho_ped))
    Z_eff_ped = core_profiles.Z_eff[ped_idx]

    Ip_MA = core_profiles.Ip_profile_face[-1] * 1e-6

    # -- Shape parameters at psi_N = _SCOPING_SHAPE_PSI_NORM --
    psi_norm_face = (psi_face - psi_face[0]) / (psi_face[-1] - psi_face[0])
    kappa_995 = jnp.interp(
        _SCOPING_SHAPE_PSI_NORM, psi_norm_face, geo.elongation_face
    )
    delta_995 = jnp.interp(
        _SCOPING_SHAPE_PSI_NORM, psi_norm_face, geo.delta_face
    )
    a_minor = geo.a_minor
    eps = a_minor / geo.R_major

    # ITER-89P cylindrical-equivalent q* (Uckan, ITER Physics Design
    # Guidelines 1989), the definition used to build the scoping training set;
    # kappa/delta are the same values passed to the network
    # (A. Ho, private communication, July 2026).
    shape_factor = (
        0.5
        * (1.0 + kappa_995**2 * (1.0 + 2.0 * delta_995**2 - 1.2 * delta_995**3))
        * (1.17 - 0.65 * eps)
        / (1.0 - eps**2) ** 2
    )
    qstar = shape_factor * 5.0 * a_minor**2 * geo.B_0 / (geo.R_major * Ip_MA)

    # Greenwald fraction with the standard Ip/(pi a^2) factor, using the
    # pedestal n_e (same definition as the sparc neped input) rather than
    # the customary line/volume average.
    fgped = formulas.calculate_greenwald_fraction(n_e_ped, core_profiles, geo)

    return {
        "Ip_MA": Ip_MA,  # [MA]
        "B_0": geo.B_0,  # [T]
        "a_minor": a_minor,  # [m]
        "aspect": geo.R_major / a_minor,  # []
        "kappa_lcfs": geo.elongation_face[-1],  # []
        "delta_lcfs": geo.delta_face[-1],  # []
        "kappa_995": kappa_995,  # []
        "delta_995": delta_995,  # []
        "qstar": qstar,  # []
        "beta_N": beta_N,  # [%]
        "Z_eff_ped": Z_eff_ped,  # []
        "n_e_ped_1e19": n_e_ped * 1e-19,  # [10^19 m^-3]
        "fgped": fgped,  # []
        "nsfrac": runtime_params.pedestal.nsfrac,  # []
        "tesep": runtime_params.pedestal.tesep,  # [eV]
    }

  def _prepare_inputs(
      self,
      runtime_params: torax.RuntimeParams,
      geo: torax.Geometry,
      core_profiles: torax.CoreProfiles,
      previous_rho_norm_ped_top: jax.Array | None = None,
  ) -> jt.Float[jt.Array, " n_inputs"]:
    """Returns this network's clipped input vector (9 sparc, 11 scoping)."""
    return self.model.pack_inputs(
        self._derive_torax_quantities(
            runtime_params, geo, core_profiles, previous_rho_norm_ped_top
        )
    )

  @override
  def _call_implementation(
      self,
      runtime_params: torax.RuntimeParams,
      geo: torax.Geometry,
      core_profiles: torax.CoreProfiles,
      pedestal_transition_state: (
          pedestal_transition_state_lib.PedestalTransitionState
      ),
  ) -> pedestal_model_output_lib.PedestalModelOutput:
    """Computes pedestal-top values from EPEDNN-mit pressure and width.

    Naming convention: In the EPED literature, "pedestal" (ψ_ped = 1 - Δ)
    and "pedestal top" (ψ_top = 1 - 1.5Δ) are distinct mtanh profile
    locations.  EPEDNN-mit outputs P_top and the width to the pedestal
    top, while TORAX I/O also uses "ped" to mean pedestal *top*
    (e.g. ``n_e_ped``, ``T_e_ped``, ``rho_norm_ped_top`` in
    ``PedestalModelOutput``).

    Within this function, local variables use ``_top`` suffixes when
    evaluated at the pedestal-top location (rho_norm_top) to distinguish from
    EPEDNN-mit inputs evaluated at ψ_ped.

    Args:
      runtime_params: Runtime parameters including pedestal config.
      geo: Geometry.
      core_profiles: Core plasma profiles.
      pedestal_transition_state: Pedestal transition state containing
        the previous pedestal model output.
    """
    assert isinstance(runtime_params.pedestal, RuntimeParams)

    # Get pedestal pressure and width from EPEDNN-mit.
    inputs = self._prepare_inputs(
        runtime_params,
        geo,
        core_profiles,
        previous_rho_norm_ped_top=pedestal_transition_state.previous_pedestal_model_output.rho_norm_ped_top,
    )
    # P_top_kPa is pressure at the pedestal top.
    # pedestal_width_psi_norm is Δψ_N from ψ=1 to the pedestal top (ψ_top).
    P_top_kPa, pedestal_width_psi_norm = self.model(inputs)

    # Convert pedestal-top width from ψ_N to ρ_N.
    psi_face = core_profiles.psi.face_value()
    psi_norm = (core_profiles.psi.value - psi_face[0]) / (
        psi_face[-1] - psi_face[0]
    )
    psi_norm_ped_top = 1.0 - pedestal_width_psi_norm
    rho_norm_ped_top = jnp.interp(psi_norm_ped_top, psi_norm, geo.rho_norm)
    rho_norm_ped_top_idx = jnp.argmin(
        jnp.abs(geo.rho_norm - rho_norm_ped_top)
    )

    # Convert pressure from kPa to Pa.
    P_top = P_top_kPa * 1e3

    # n_e_top: pedestal-top electron density (TORAX API names this "n_e_ped").
    n_e_top = runtime_params.pedestal.n_e_ped

    # -- Calculate T_e and T_i at the pedestal top from P_top --
    # Evaluate composition at the pedestal-top location.
    temperature_ratio = runtime_params.pedestal.T_i_T_e_ratio
    Z_eff_top = core_profiles.Z_eff[rho_norm_ped_top_idx]
    Z_i_top = core_profiles.Z_i[rho_norm_ped_top_idx]
    Z_impurity_top = core_profiles.Z_impurity[rho_norm_ped_top_idx]
    dilution_factor_top = jnp.where(
        Z_eff_top == 1.0,
        1.0,
        formulas.calculate_main_ion_dilution_factor(
            Z_i_top, Z_impurity_top, Z_eff_top
        ),
    )
    n_i_top = dilution_factor_top * n_e_top
    safe_Z_impurity_top = jnp.where(
        Z_eff_top == 1.0, 1.0, Z_impurity_top
    )
    n_impurity_top = jnp.where(
        Z_eff_top == 1.0,
        0.0,
        (n_e_top - Z_i_top * n_i_top) / safe_Z_impurity_top,
    )
    # P = T_e*n_e + T_i*n_i + T_i*n_imp  (assuming T_imp = T_i)
    T_e_top = (
        P_top * runtime_params.pedestal.P_ped_multiplier
    ) / (
        n_e_top
        + temperature_ratio * n_i_top
        + temperature_ratio * n_impurity_top
    ) / constants.CONSTANTS.keV_to_J
    T_i_top = temperature_ratio * T_e_top

    # TORAX API uses "ped" to mean pedestal top.
    return pedestal_model_output_lib.PedestalModelOutput(
        n_e_ped=n_e_top,
        T_i_ped=T_i_top,
        T_e_ped=T_e_top,
        rho_norm_ped_top=rho_norm_ped_top,
    )


# TODO: Keep aligned with upcoming TORAX V2 API changes where n_e_ped will be
# renamed to n_e_top for consistency with EPED naming conventions.
class EPEDNNmitConfig(torax.pedestal.BasePedestal):
  """TORAX pedestal model config using EPEDNN-mit.

  Attributes:
    machine: Which EPEDNN-mit network to use ("sparc" or "scoping").
    n_e_ped: The electron density at the pedestal top [m^-3].
    T_i_T_e_ratio: Ratio of the ion and electron temperature at the pedestal
      [dimensionless].
    P_ped_multiplier: Multiplier for the pedestal pressure (mostly used for
      sensitivity analysis) [dimensionless].
    nsfrac: Separatrix-to-pedestal electron density ratio n_e_sep / n_e_ped
      [dimensionless]. Scoping-model input; unused by "sparc".
    tesep: Separatrix electron temperature [eV]. Scoping-model input; unused
      by "sparc".
  """

  model_name: Annotated[Literal["epednn_mit"], torax_pydantic.JAX_STATIC] = (
      "epednn_mit"
  )
  machine: Annotated[EPEDNNmitMachine, torax_pydantic.JAX_STATIC] = "sparc"
  n_e_ped: torax_pydantic.TimeVaryingScalar = torax_pydantic.ValidatedDefault(
      0.7e20
  )
  T_i_T_e_ratio: torax_pydantic.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(1.0)
  )
  P_ped_multiplier: torax_pydantic.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(1.0)
  )
  nsfrac: torax_pydantic.TimeVaryingScalar = torax_pydantic.ValidatedDefault(
      0.4
  )
  tesep: torax_pydantic.TimeVaryingScalar = torax_pydantic.ValidatedDefault(
      75.0
  )

  def build_pedestal_model(
      self,
  ) -> EPEDNNmitPedestalModel:
    return EPEDNNmitPedestalModel(
        machine=self.machine,
        formation_model=self.formation_model.build_formation_model(),
        saturation_model=self.saturation_model.build_saturation_model(),
    )

  def build_runtime_params(self, t: chex.Numeric) -> RuntimeParams:
    base_runtime_params = super().build_runtime_params(t)
    # Rebuild the base fields generically so this survives base-class field
    # additions and renames across TORAX versions (e.g. the
    # use_formation_model_* rename between 1.4.2 and head).
    base_fields = {
        field.name: getattr(base_runtime_params, field.name)
        for field in dataclasses.fields(type(base_runtime_params))
    }
    return RuntimeParams(
        **base_fields,
        P_ped_multiplier=self.P_ped_multiplier.get_value(t),
        n_e_ped=self.n_e_ped.get_value(t),
        T_i_T_e_ratio=self.T_i_T_e_ratio.get_value(t),
        nsfrac=self.nsfrac.get_value(t),
        tesep=self.tesep.get_value(t),
    )
