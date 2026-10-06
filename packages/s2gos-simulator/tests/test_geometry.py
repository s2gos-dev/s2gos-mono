from datetime import datetime, timezone
from pathlib import Path

import boto3
import pystac_client
import pytest
from s2gos_utils.setting.credentials.credential import S3Credential
from s2gos_utils.setting.credentials.provider import (
    DictCredentialProvider,
    set_credential_provider,
)

from s2gos_simulator.config.geometry import (
    fetch_tile_metadata,
    platform_from_tile_metadata,
)
from s2gos_simulator.config.sensors import SatellitePlatform

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_mtd_tl(path: Path, tile_id: str, sensing_time: str = "2022-05-04T09:17:15.458038Z"):
    path.write_text(
        f"""<?xml version="1.0" encoding="UTF-8"?>
<n1:Level-1C_Tile_ID xmlns:n1="https://psd-14.sentinel2.eo.esa.int/PSD/S2_PDI_Level-1C_Tile_Metadata.xsd">
  <n1:General_Info>
    <TILE_ID metadataLevel="Brief">{tile_id}</TILE_ID>
    <SENSING_TIME metadataLevel="Standard">{sensing_time}</SENSING_TIME>
  </n1:General_Info>
</n1:Level-1C_Tile_ID>
"""
    )
    return path


class _FakeAsset:
    def __init__(self, href):
        self.href = href


class _FakeItem:
    def __init__(self, dt, href):
        self.datetime = dt
        self.assets = {"granule_metadata": _FakeAsset(href)}


class _FakeSearch:
    def __init__(self, items):
        self._items = items

    def items(self):
        return iter(self._items)


class _FakeCatalog:
    def __init__(self, items, search_calls):
        self._items = items
        self._search_calls = search_calls

    def add_conforms_to(self, _conformance_class):
        pass

    def search(self, **kwargs):
        self._search_calls.append(kwargs)
        return _FakeSearch(self._items)


class _FakeS3Client:
    def __init__(self, download_calls, **init_kwargs):
        self.init_kwargs = init_kwargs
        self._download_calls = download_calls

    def download_file(self, bucket, key, dest):
        self._download_calls.append((bucket, key, dest))
        Path(dest).write_text("<fake-mtd-tl/>")


def _patch_stac_and_s3(monkeypatch, items):
    """Patches pystac_client.Client.open and boto3.client with recording fakes.

    Returns (search_calls, download_calls, client_init_kwargs) — lists/dict
    populated once fetch_tile_metadata runs, for assertions.
    """
    search_calls = []
    download_calls = []
    client_init_kwargs = {}

    catalog = _FakeCatalog(items, search_calls)
    monkeypatch.setattr(pystac_client.Client, "open", lambda url: catalog)

    def fake_boto3_client(service_name, **kwargs):
        assert service_name == "s3"
        client_init_kwargs.update(kwargs)
        return _FakeS3Client(download_calls, **kwargs)

    monkeypatch.setattr(boto3, "client", fake_boto3_client)

    return search_calls, download_calls, client_init_kwargs


def _set_fake_credential(credential_id="fake_s3", endpoint_url="https://s3.example.com"):
    cred = S3Credential(id=credential_id, key="k", secret="s", endpoint_url=endpoint_url)
    set_credential_provider(DictCredentialProvider({credential_id: cred}))


# ---------------------------------------------------------------------------
# platform_from_tile_metadata
# ---------------------------------------------------------------------------


class TestPlatformFromTileMetadata:
    def test_detects_sentinel_2a(self, tmp_path):
        path = _write_mtd_tl(
            tmp_path / "MTD_TL.xml",
            "S2A_OPER_MSI_L1C_TL_S2RP_20220504T090532_A035854_T33KWP_N05.10",
        )
        assert platform_from_tile_metadata(path) == SatellitePlatform.SENTINEL_2A

    def test_detects_sentinel_2b(self, tmp_path):
        path = _write_mtd_tl(
            tmp_path / "MTD_TL.xml",
            "S2B_OPER_MSI_L1C_TL_S2RP_20220509T091458_A027017_T33KWQ_N05.10",
        )
        assert platform_from_tile_metadata(path) == SatellitePlatform.SENTINEL_2B

    def test_unrecognized_prefix_raises(self, tmp_path):
        path = _write_mtd_tl(tmp_path / "MTD_TL.xml", "S2C_OPER_MSI_L1C_TL_...")
        with pytest.raises(ValueError, match="Unrecognized Sentinel-2 platform"):
            platform_from_tile_metadata(path)


# ---------------------------------------------------------------------------
# fetch_tile_metadata
# ---------------------------------------------------------------------------


class TestFetchTileMetadata:
    def test_cache_hit_skips_network_entirely(self, tmp_path, monkeypatch):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        cached = _write_mtd_tl(
            cache_dir / "MTD_TL_Gobabeb_20220504.xml",
            "S2A_OPER_MSI_L1C_TL_S2RP_20220504T090532_A035854_T33KWP_N05.10",
        )

        def fail(*args, **kwargs):
            raise AssertionError("network should not be touched on a cache hit")

        monkeypatch.setattr(pystac_client.Client, "open", fail)
        monkeypatch.setattr(boto3, "client", fail)

        result = fetch_tile_metadata(
            "2022-05-04", -23.600, 15.11956, cache_dir, site_name="Gobabeb"
        )
        assert result == cached

    def test_downloads_and_caches_on_miss(self, tmp_path, monkeypatch):
        anchor = datetime(2022, 5, 4, tzinfo=timezone.utc)
        item = _FakeItem(
            anchor, "s3://eodata/Sentinel-2/MSI/L1C/2022/05/04/x.SAFE/GRANULE/g1/MTD_TL.xml"
        )
        search_calls, download_calls, client_kwargs = _patch_stac_and_s3(monkeypatch, [item])
        _set_fake_credential()

        cache_dir = tmp_path / "cache"
        result = fetch_tile_metadata(
            "2022-05-04",
            -23.600,
            15.11956,
            cache_dir,
            site_name="Gobabeb",
            credential_id="fake_s3",
        )

        assert result == cache_dir / "MTD_TL_Gobabeb_20220504.xml"
        assert result.exists()
        assert download_calls == [
            ("eodata", "Sentinel-2/MSI/L1C/2022/05/04/x.SAFE/GRANULE/g1/MTD_TL.xml", str(result))
        ]
        # intersects must be (lon, lat) GeoJSON order, not (lat, lon).
        assert search_calls[0]["intersects"] == {
            "type": "Point",
            "coordinates": [15.11956, -23.600],
        }
        assert search_calls[0]["collections"] == ["sentinel-2-l1c"]

    def test_endpoint_url_with_scheme_is_not_doubled(self, tmp_path, monkeypatch):
        """Regression test: a credential endpoint_url that already carries a
        scheme (the documented .secrets.yaml convention) must not become
        'https://https://...'."""
        anchor = datetime(2022, 5, 4, tzinfo=timezone.utc)
        item = _FakeItem(anchor, "s3://eodata/some/key/MTD_TL.xml")
        _, _, client_kwargs = _patch_stac_and_s3(monkeypatch, [item])
        _set_fake_credential(endpoint_url="https://s3.de.io.cloud.ovh.net")

        fetch_tile_metadata(
            "2022-05-04", -23.600, 15.11956, tmp_path / "cache", credential_id="fake_s3"
        )

        assert client_kwargs["endpoint_url"] == "https://s3.de.io.cloud.ovh.net"

    def test_endpoint_url_without_scheme_gets_https_prefixed(self, tmp_path, monkeypatch):
        anchor = datetime(2022, 5, 4, tzinfo=timezone.utc)
        item = _FakeItem(anchor, "s3://eodata/some/key/MTD_TL.xml")
        _, _, client_kwargs = _patch_stac_and_s3(monkeypatch, [item])
        _set_fake_credential(endpoint_url="eodata.dataspace.copernicus.eu")

        fetch_tile_metadata(
            "2022-05-04", -23.600, 15.11956, tmp_path / "cache", credential_id="fake_s3"
        )

        assert client_kwargs["endpoint_url"] == "https://eodata.dataspace.copernicus.eu"

    def test_picks_item_closest_to_anchor_date(self, tmp_path, monkeypatch):
        far = _FakeItem(
            datetime(2022, 5, 9, tzinfo=timezone.utc), "s3://eodata/far/MTD_TL.xml"
        )
        near = _FakeItem(
            datetime(2022, 5, 5, tzinfo=timezone.utc), "s3://eodata/near/MTD_TL.xml"
        )
        _, download_calls, _ = _patch_stac_and_s3(monkeypatch, [far, near])
        _set_fake_credential()

        fetch_tile_metadata(
            "2022-05-04", -23.600, 15.11956, tmp_path / "cache", credential_id="fake_s3"
        )

        assert download_calls[0][0:2] == ("eodata", "near/MTD_TL.xml")

    def test_raises_when_no_tile_found(self, tmp_path, monkeypatch):
        _patch_stac_and_s3(monkeypatch, [])
        _set_fake_credential()

        with pytest.raises(RuntimeError, match="No Sentinel-2 L1C tile found"):
            fetch_tile_metadata(
                "2022-05-04", -23.600, 15.11956, tmp_path / "cache", credential_id="fake_s3"
            )
