"""Tests for the Torax EPEDNN-mit interface."""

import dataclasses
from absl.testing import absltest
from absl.testing import parameterized
from epednn_mit.interfaces import torax_interface as epednn_mit_torax_interface
import jax
from jax import numpy as jnp
import numpy as np
import torax
from torax._src.pedestal_model import pedestal_transition_state as pedestal_transition_state_lib

# pylint: disable=invalid-name


class PackedInputsContract:
  """Shared assertions about input packing, run for every machine.

  Mixed into the per-machine test cases, which supply ``pedestal_model``,
  ``runtime_params``, ``geo`` and ``core_profiles``.
  """

  def test_packed_inputs_match_named_sources(self):
    """Each packed element is the named quantity, clipped to its own range.

    This is what ties the input vector to the clip bounds: a packing order
    that disagreed with the declared ranges would clip inputs against each
    other's limits, which is silent in the prediction.
    """
    model = self.pedestal_model.model
    quantities = self.pedestal_model._derive_torax_quantities(  # pylint: disable=protected-access
        self.runtime_params, self.geo, self.core_profiles, None
    )
    packed = model.pack_inputs(quantities)
    available = quantities | model.constants

    self.assertLen(packed, len(model.input_specs))
    for i, (name, (source, lo, hi)) in enumerate(model.input_specs.items()):
      np.testing.assert_allclose(
          packed[i],
          np.clip(np.float32(available[source]), lo, hi),
          rtol=1e-6,
          err_msg=f'input {name} (from {source})',
      )


class EPEDNNmitPedestalModelTest(PackedInputsContract, parameterized.TestCase):

  def setUp(self):
    super().setUp()

    # Register the EPEDNN-mit pedestal model with TORAX.
    torax.pedestal.register_pedestal_model(
        epednn_mit_torax_interface.EPEDNNmitConfig
    )

    # Create a TORAX config with the EPEDNN-mit pedestal model.
    config = {
        'profile_conditions': {},
        'plasma_composition': {},
        'numerics': {},
        'geometry': {'geometry_type': 'circular', 'n_rho': 4},
        'sources': {},
        'solver': {},
        'transport': {},
        'pedestal': {
            'model_name': 'epednn_mit',
            'set_pedestal': True,
            'n_e_ped': 0.7e20,
            'T_i_T_e_ratio': 1.0,
        },
    }
    torax_config = torax.ToraxConfig.from_dict(config)
    self.pedestal_model = torax_config.pedestal.build_pedestal_model()
    self.jitted_pedestal_model = jax.jit(self.pedestal_model)
    step_fn = torax.experimental.make_step_fn(torax_config)
    state, _ = torax.experimental.get_initial_state_and_post_processed_outputs(
        step_fn
    )
    self.runtime_params = step_fn.runtime_params_provider(t=0.0)
    self.geo = state.geometry
    self.core_profiles = state.core_profiles
    self.source_profiles = state.core_sources

  def test_build_and_call_pedestal_model(self):
    """Tests the EPEDNN-mit pedestal model.

    Note that the EPEDNN-mit is only valid for SPARC parameter space, but we're
    testing here with a generic config. Hence, we don't perform checks on
    the values of the model predictions, but only that the values set in the
    config are passed through to the output.
    """
    assert isinstance(
        self.runtime_params.pedestal, epednn_mit_torax_interface.RuntimeParams
    )
    assert isinstance(
        self.pedestal_model, epednn_mit_torax_interface.EPEDNNmitPedestalModel
    )

    pedestal_model_output = self.jitted_pedestal_model(
        runtime_params=self.runtime_params,
        geo=self.geo,
        core_profiles=self.core_profiles,
        source_profiles=self.source_profiles,
        pedestal_transition_state=pedestal_transition_state_lib.PedestalTransitionState.empty_L_mode(),
    )

    # These values come from the config, not the model.
    np.testing.assert_allclose(pedestal_model_output.n_e_ped, 0.7e20)
    np.testing.assert_allclose(
        pedestal_model_output.T_i_ped / pedestal_model_output.T_e_ped, 1.0
    )

  def test_pedestal_pressure_multiplier(self):
    """Tests that P_ped_multiplier scales the temperature outputs."""
    assert isinstance(
        self.runtime_params.pedestal, epednn_mit_torax_interface.RuntimeParams
    )
    assert isinstance(
        self.pedestal_model, epednn_mit_torax_interface.EPEDNNmitPedestalModel
    )

    # Get outputs with multiplier 1.0
    pedestal_model_output_1 = self.jitted_pedestal_model(
        runtime_params=self.runtime_params,
        geo=self.geo,
        core_profiles=self.core_profiles,
        source_profiles=self.source_profiles,
        pedestal_transition_state=pedestal_transition_state_lib.PedestalTransitionState.empty_L_mode(),
    )

    # Replace the multiplier in runtime params and re-run
    modified_runtime_params = dataclasses.replace(
        self.runtime_params,
        pedestal=dataclasses.replace(
            self.runtime_params.pedestal,
            P_ped_multiplier=1.5,
        ),
    )

    pedestal_model_output_2 = self.jitted_pedestal_model(
        runtime_params=modified_runtime_params,
        geo=self.geo,
        core_profiles=self.core_profiles,
        source_profiles=self.source_profiles,
        pedestal_transition_state=pedestal_transition_state_lib.PedestalTransitionState.empty_L_mode(),
    )

    np.testing.assert_allclose(
        pedestal_model_output_2.T_e_ped, pedestal_model_output_1.T_e_ped * 1.5
    )
    np.testing.assert_allclose(
        pedestal_model_output_2.T_i_ped, pedestal_model_output_1.T_i_ped * 1.5
    )
    np.testing.assert_allclose(
        pedestal_model_output_2.n_e_ped, pedestal_model_output_1.n_e_ped
    )


class EPEDNNmitScopingPedestalModelTest(
    PackedInputsContract, parameterized.TestCase
):

  def setUp(self):
    super().setUp()

    torax.pedestal.register_pedestal_model(
        epednn_mit_torax_interface.EPEDNNmitConfig
    )

    config = {
        'profile_conditions': {},
        'plasma_composition': {},
        'numerics': {},
        'geometry': {'geometry_type': 'circular', 'n_rho': 4},
        'sources': {},
        'solver': {},
        'transport': {},
        'pedestal': {
            'model_name': 'epednn_mit',
            'machine': 'scoping',
            'set_pedestal': True,
            'n_e_ped': 0.7e20,
            'T_i_T_e_ratio': 1.0,
            'nsfrac': 0.42,
            'tesep': 100.0,
        },
    }
    torax_config = torax.ToraxConfig.from_dict(config)
    self.pedestal_model = torax_config.pedestal.build_pedestal_model()
    self.jitted_pedestal_model = jax.jit(self.pedestal_model)
    step_fn = torax.experimental.make_step_fn(torax_config)
    state, _ = torax.experimental.get_initial_state_and_post_processed_outputs(
        step_fn
    )
    self.runtime_params = step_fn.runtime_params_provider(t=0.0)
    self.geo = state.geometry
    self.core_profiles = state.core_profiles
    self.source_profiles = state.core_sources

  def test_machine_reaches_the_model(self):
    """The config machine selects the scoping network (not the default sparc)."""
    assert isinstance(
        self.pedestal_model, epednn_mit_torax_interface.EPEDNNmitPedestalModel
    )
    self.assertEqual(self.pedestal_model.machine, 'scoping')
    self.assertIsInstance(
        self.pedestal_model.model, epednn_mit_torax_interface.ScopingModel
    )
    self.assertLen(self.pedestal_model.model.input_specs, 11)

  def test_build_and_call_pedestal_model(self):
    """Builds and calls the scoping pedestal model on a generic config.

    As with the sparc test, the generic circular test config is not a
    physical scoping-space point, so only pass-through values and output
    sanity are checked.
    """
    assert isinstance(
        self.runtime_params.pedestal, epednn_mit_torax_interface.RuntimeParams
    )
    np.testing.assert_allclose(self.runtime_params.pedestal.nsfrac, 0.42)
    np.testing.assert_allclose(self.runtime_params.pedestal.tesep, 100.0)

    pedestal_model_output = self.jitted_pedestal_model(
        runtime_params=self.runtime_params,
        geo=self.geo,
        core_profiles=self.core_profiles,
        source_profiles=self.source_profiles,
        pedestal_transition_state=pedestal_transition_state_lib.PedestalTransitionState.empty_L_mode(),
    )

    np.testing.assert_allclose(pedestal_model_output.n_e_ped, 0.7e20)
    np.testing.assert_allclose(
        pedestal_model_output.T_i_ped / pedestal_model_output.T_e_ped, 1.0
    )
    self.assertTrue(np.isfinite(pedestal_model_output.T_e_ped))
    self.assertGreater(pedestal_model_output.T_e_ped, 0.0)
    self.assertBetween(float(pedestal_model_output.rho_norm_ped_top), 0.0, 1.0)

  def test_scoping_wrapper_matches_reference_vector(self):
    """The wrapper reproduces the scoping network's own reference vector.

    Input and expected output are those of
    ``models.scoping.jax_model.test_epednn_mit_scoping_jax``.
    """
    wrapper = epednn_mit_torax_interface.ScopingModel()
    test_in = jnp.array(
        [1.3, 3.0, 1.9, 0.5, 8.7, 4.0, 2.0, 2.5, 0.75, 0.5, 300.0],
        dtype=jnp.float32,
    )
    P_top_kPa, pedestal_width_psi_norm = wrapper(test_in)
    np.testing.assert_allclose(P_top_kPa, 3.5654504e02, rtol=1e-4)
    np.testing.assert_allclose(
        pedestal_width_psi_norm, 5.6555986e-02, rtol=1e-4
    )


class EPEDNNmitNetworkTest(parameterized.TestCase):
  """Tests over the network wrappers themselves, independent of TORAX state."""

  def test_sparc_wrapper_matches_reference_vector(self):
    """The wrapper reproduces the sparc network's own reference vector.

    Input and expected output are those of
    ``models.sparc.jax_model.test_epednn_mit_sparc_jax``, so a change here
    means the interface disagrees with the network it wraps.

    ``pack_inputs`` supplies the input at the precision the network requires
    (see its dtype note). These values sit inside the training ranges, so the
    clip is a no-op.
    """
    wrapper = epednn_mit_torax_interface.SparcModel()
    test_in = wrapper.pack_inputs({
        'Ip_MA': 10.0,
        'B_0': 9.0,
        'kappa_lcfs': 2.0,
        'delta_lcfs': 0.45,
        'n_e_ped_1e19': 30.0,
        'beta_N': 1.2,
        'Z_eff_ped': 2.0,
    })
    P_top_kPa, pedestal_width_psi_norm = wrapper(test_in)
    np.testing.assert_allclose(P_top_kPa, 3.19934668e02, rtol=1e-4)
    np.testing.assert_allclose(
        pedestal_width_psi_norm, 4.11556657e-02, rtol=1e-4
    )

  @parameterized.parameters(('sparc', 9), ('scoping', 11))
  def test_input_width(self, machine, expected_width):
    """Each network declares exactly as many inputs as it was trained on."""
    wrapper = epednn_mit_torax_interface.MODELS[machine]()
    self.assertLen(wrapper.input_specs, expected_width)
    self.assertLen(wrapper.input_lower_bounds, expected_width)
    self.assertLen(wrapper.input_upper_bounds, expected_width)

  @parameterized.parameters('sparc', 'scoping')
  def test_bounds_are_ordered(self, machine):
    wrapper = epednn_mit_torax_interface.MODELS[machine]()
    for name, (_, lo, hi) in wrapper.input_specs.items():
      self.assertLessEqual(lo, hi, msg=f'{machine} input {name}')

  def test_out_of_range_inputs_are_named(self):
    """Clipped inputs are reported by name, not just silently clamped."""
    wrapper = epednn_mit_torax_interface.ScopingModel()
    # Every input at 1.0: below qstar's (3.0, 5.0) but inside fgped's
    # (0.3, 1.3).
    quantities = {
        source: 1.0 for source, _, _ in wrapper.input_specs.values()
    }
    out_of_range = wrapper.out_of_range_inputs(quantities)
    self.assertEqual(out_of_range['qstar'], (1.0, 3.0, 5.0))
    self.assertNotIn('fgped', out_of_range)

  def test_unknown_machine_is_rejected_by_the_config(self):
    """An unsupported machine fails config validation, not at run time."""
    torax.pedestal.register_pedestal_model(
        epednn_mit_torax_interface.EPEDNNmitConfig
    )
    with self.assertRaises(ValueError):
      epednn_mit_torax_interface.EPEDNNmitConfig.from_dict(
          {'model_name': 'epednn_mit', 'machine': 'not_a_machine'}
      )


if __name__ == '__main__':
  absltest.main()
