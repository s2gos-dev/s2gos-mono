import pytest

from s2gos_simulator.config import INSTRUMENTS, MultispectralInstrument


def _eradiate_srf_ids():
    for key, instrument in INSTRUMENTS.items():
        srfs = (
            [band.srf for band in instrument.bands.values()]
            if isinstance(instrument, MultispectralInstrument)
            else [instrument.srf]
        )
        yield from ((key, srf) for srf in srfs if isinstance(srf, str))


@pytest.mark.filterwarnings("ignore:Initializing a BandSRF")
@pytest.mark.parametrize("key, srf_id", list(_eradiate_srf_ids()))
def test_catalogue_srf_ids_exist_in_eradiate(key, srf_id):
    response = pytest.importorskip("eradiate.spectral.response")

    response.BandSRF.from_id(srf_id)
