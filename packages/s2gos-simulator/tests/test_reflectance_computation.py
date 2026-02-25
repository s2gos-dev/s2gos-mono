import numpy as np
import pytest
import xarray as xr

from s2gos_simulator.backends.eradiate.reflectance_computation import (
    align_and_broadcast_datasets,
    compute_bhr,
    compute_brf,
    compute_hcrf,
    compute_hdrf,
)


def _scalar_da(value, wavelengths=None):
    if wavelengths is not None:
        data = np.full(len(wavelengths), float(value))
        return xr.DataArray(data, dims=["w"], coords={"w": wavelengths})
    return xr.DataArray(float(value))


class TestComputeBRF:
    def test_brf_formula_known_values_cos_sza_1(self):
        # BRF = (π × L) / (E_toa × cos_sza)
        L = _scalar_da(1.0 / np.pi)
        E = _scalar_da(1.0)
        result = compute_brf(L, E, cos_sza=1.0, measurement_id="test")
        np.testing.assert_allclose(float(result["brf"]), 1.0)

    def test_brf_formula_known_values_cos_sza_half(self):
        L = _scalar_da(1.0 / np.pi)
        E = _scalar_da(1.0)
        result = compute_brf(L, E, cos_sza=0.5, measurement_id="test")
        np.testing.assert_allclose(float(result["brf"]), 2.0)

    def test_brf_attributes(self):
        L = _scalar_da(1.0 / np.pi, wavelengths=[550.0])
        E = _scalar_da(1.0, wavelengths=[550.0])
        cos_sza = 0.7
        result = compute_brf(L, E, cos_sza=cos_sza, measurement_id="my_brf")
        assert "brf" in result.data_vars
        assert result.attrs["measurement_type"] == "brf"
        assert result.attrs["cos_sza"] == cos_sza
        assert "E_toa" in result.attrs["formula"]

    def test_extra_attrs_merged_brf(self):
        L = _scalar_da(1.0 / np.pi)
        E = _scalar_da(1.0)
        result = compute_brf(
            L, E, cos_sza=1.0, measurement_id="brf", extra_attrs={"scene": "test"}
        )
        assert result.attrs["scene"] == "test"


class TestComputeHDRF:
    def test_hdrf_formula_known_values_unit(self):
        # HDRF = (π × L) / E_boa
        L = _scalar_da(1.0 / np.pi)
        E = _scalar_da(1.0)
        result = compute_hdrf(L, E, measurement_id="test")
        np.testing.assert_allclose(float(result["hdrf"]), 1.0)

    def test_hdrf_formula_known_values_double(self):
        L = _scalar_da(2.0 / np.pi)
        E = _scalar_da(2.0)
        result = compute_hdrf(L, E, measurement_id="test")
        np.testing.assert_allclose(float(result["hdrf"]), 1.0)

    def test_hdrf_variable_name(self):
        L = _scalar_da(1.0 / np.pi)
        E = _scalar_da(1.0)
        result = compute_hdrf(L, E, measurement_id="m1")
        assert "hdrf" in result.data_vars

    def test_hdrf_measurement_type_attr(self):
        L = _scalar_da(1.0 / np.pi)
        E = _scalar_da(1.0)
        result = compute_hdrf(L, E, measurement_id="m1")
        assert result.attrs["measurement_type"] == "hdrf"


class TestComputeHCRF:
    def test_hcrf_formula_known_values(self):
        L = _scalar_da(1.0 / np.pi)
        E = _scalar_da(1.0)
        result = compute_hcrf(L, E, measurement_id="test")
        np.testing.assert_allclose(float(result["hcrf"]), 1.0)

    def test_hcrf_variable_name(self):
        L = _scalar_da(1.0 / np.pi)
        E = _scalar_da(1.0)
        result = compute_hcrf(L, E, measurement_id="m1")
        assert "hcrf" in result.data_vars

    def test_hcrf_measurement_type_attr(self):
        L = _scalar_da(1.0 / np.pi)
        E = _scalar_da(1.0)
        result = compute_hcrf(L, E, measurement_id="m1")
        assert result.attrs["measurement_type"] == "hcrf"


class TestHDRFvsHCRF:
    def test_hdrf_hcrf_formulas_identical_values(self):
        """HDRF and HCRF use the same formula; only variable name differs."""
        wl = [500.0, 600.0, 700.0]
        L = xr.DataArray([0.1, 0.2, 0.3], dims=["w"], coords={"w": wl})
        E = xr.DataArray([1.0, 1.0, 1.0], dims=["w"], coords={"w": wl})
        hdrf = compute_hdrf(L, E, measurement_id="m")
        hcrf = compute_hcrf(L, E, measurement_id="m")
        np.testing.assert_array_equal(hdrf["hdrf"].values, hcrf["hcrf"].values)

    def test_hdrf_hcrf_different_variable_names(self):
        L = _scalar_da(1.0 / np.pi)
        E = _scalar_da(1.0)
        hdrf = compute_hdrf(L, E, measurement_id="m")
        hcrf = compute_hcrf(L, E, measurement_id="m")
        assert "hdrf" in hdrf.data_vars
        assert "hcrf" in hcrf.data_vars
        assert "hcrf" not in hdrf.data_vars
        assert "hdrf" not in hcrf.data_vars

    def test_extra_attrs_merged(self):
        L = _scalar_da(1.0 / np.pi)
        E = _scalar_da(1.0)
        result = compute_hdrf(L, E, measurement_id="m", extra_attrs={"scene": "test"})
        assert result.attrs["scene"] == "test"


class TestComputeBHR:
    def test_bhr_formula_known_values(self):
        # BHR = J_surface / J_reference
        J = _scalar_da(0.6)
        J_ref = _scalar_da(1.0)
        result = compute_bhr(J, J_ref, measurement_id="test")
        np.testing.assert_allclose(float(result["bhr"]), 0.6)

    def test_bhr_variable_name(self):
        J = _scalar_da(0.5)
        J_ref = _scalar_da(1.0)
        result = compute_bhr(J, J_ref, measurement_id="m")
        assert "bhr" in result.data_vars

    def test_bhr_measurement_type_attr(self):
        J = _scalar_da(0.5)
        J_ref = _scalar_da(1.0)
        result = compute_bhr(J, J_ref, measurement_id="m")
        assert result.attrs["measurement_type"] == "bhr"


class TestAlignAndBroadcast:
    def test_align_same_wavelengths_unchanged(self):
        wl = [500.0, 600.0, 700.0]
        rad = xr.DataArray([1.0, 2.0, 3.0], dims=["w"], coords={"w": wl})
        ref = xr.DataArray([10.0, 20.0, 30.0], dims=["w"], coords={"w": wl})
        rad_out, ref_out = align_and_broadcast_datasets(rad, ref)
        np.testing.assert_array_equal(rad_out.values, rad.values)
        np.testing.assert_array_equal(ref_out.values, ref.values)

    def test_align_interpolates_reference_to_radiance_grid(self):
        rad_wl = [500.0, 600.0, 700.0]
        ref_wl = [400.0, 500.0, 600.0, 700.0, 800.0]
        rad = xr.DataArray([1.0, 2.0, 3.0], dims=["w"], coords={"w": rad_wl})
        ref = xr.DataArray([0.5, 1.0, 1.5, 2.0, 2.5], dims=["w"], coords={"w": ref_wl})
        _, ref_out = align_and_broadcast_datasets(rad, ref)

        assert list(ref_out["w"].values) == rad_wl
        np.testing.assert_allclose(ref_out.values, [1.0, 1.5, 2.0], atol=1e-10)
        assert not np.any(np.isnan(ref_out.values))

    def test_align_raises_when_radiance_extends_beyond_reference(self):
        rad = xr.DataArray([1.0, 2.0], dims=["w"], coords={"w": [400.0, 900.0]})
        ref = xr.DataArray(
            [1.0, 1.0, 1.0], dims=["w"], coords={"w": [500.0, 600.0, 700.0]}
        )
        with pytest.raises(ValueError, match="extends beyond"):
            align_and_broadcast_datasets(rad, ref)

    def test_align_fill_value_constant_fills_out_of_range(self):
        rad = xr.DataArray([1.0, 2.0], dims=["w"], coords={"w": [400.0, 900.0]})
        ref = xr.DataArray(
            [1.0, 1.0, 1.0], dims=["w"], coords={"w": [500.0, 600.0, 700.0]}
        )
        _, ref_out = align_and_broadcast_datasets(rad, ref, fill_value=0.0)
        assert ref_out.sel(w=400.0).item() == pytest.approx(0.0)
        assert ref_out.sel(w=900.0).item() == pytest.approx(0.0)
