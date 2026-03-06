import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from pydantic import ValidationError

from s2gos_generator.core.config.assets import UserAssets, XmlSceneConfig

_PATCH_RESOLVE = "s2gos_generator.core.config.assets._resolve_asset_path"
_PATCH_MKDIR = "s2gos_utils.io.paths.mkdir"
_PATCH_COPY = "s2gos_utils.io.paths.copy"
_PATCH_OPEN_FILE = "s2gos_generator.resources.assets.open_file"


class TestUserAssetsValidation:
    """Tests for UserAssets field validators via model_validate."""

    def _valid_data(self, **overrides):
        data = dict(
            object_id="tree",
            ply_path="/fake/mesh.ply",
            coordinate=[5.0, 10.0],
            coord_type="scene",
            material="diffuse",
        )
        data.update(overrides)
        return data

    def _validate(self, data):
        with patch(_PATCH_RESOLVE, side_effect=lambda v, **kw: v):
            return UserAssets.model_validate(data)

    def test_coordinate_must_have_two_elements(self):
        with pytest.raises(ValidationError, match="Coordinate must be"):
            self._validate(self._valid_data(coordinate=[1.0, 2.0, 3.0]))

    def test_geographic_longitude_bounds(self):
        with pytest.raises(ValidationError, match="Longitude"):
            self._validate(
                self._valid_data(coordinate=[200.0, 45.0], coord_type="geographic")
            )

    def test_geographic_latitude_bounds(self):
        with pytest.raises(ValidationError, match="Latitude"):
            self._validate(
                self._valid_data(coordinate=[10.0, 100.0], coord_type="geographic")
            )

    def test_scale_must_be_positive(self):
        with pytest.raises(ValidationError, match="Scale must be positive"):
            self._validate(self._valid_data(scale=0.0))

    def test_material_string_must_be_nonempty(self):
        with pytest.raises(ValidationError, match="Material reference cannot be empty"):
            self._validate(self._valid_data(material="   "))

    def test_material_dict_must_have_type_key(self):
        with pytest.raises(ValidationError, match="'type' field"):
            self._validate(self._valid_data(material={"reflectance": 0.5}))

    def test_inline_material_helpers(self):
        asset = UserAssets.model_construct(
            object_id="rock",
            material={"type": "diffuse", "reflectance": 0.3},
        )
        assert asset.get_inline_material_id() == "rock_material"
        assert asset.get_inline_material_dict() == {
            "type": "diffuse",
            "reflectance": 0.3,
        }


class TestXmlSceneConfigValidation:
    """Tests for XmlSceneConfig field validators via model_validate."""

    def _valid_data(self, **overrides):
        data = dict(
            xml_path="/fake/scene.xml",
            base_coordinate=(5.0, 10.0),
            coord_type="scene",
        )
        data.update(overrides)
        return data

    def _validate(self, data):
        with patch(_PATCH_RESOLVE, side_effect=lambda v, **kw: v):
            return XmlSceneConfig.model_validate(data)

    def test_base_coordinate_must_have_two_elements(self):
        with pytest.raises(
            ValidationError, match="at most 2 items|Base coordinate must be"
        ):
            self._validate(self._valid_data(base_coordinate=(1.0, 2.0, 3.0)))

    def test_scale_must_be_positive(self):
        with pytest.raises(ValidationError, match="greater than"):
            self._validate(self._valid_data(scale=0.0))

    def test_geographic_coordinate_bounds(self):
        with pytest.raises(ValidationError, match="Longitude"):
            self._validate(
                self._valid_data(base_coordinate=(200.0, 45.0), coord_type="geographic")
            )


class TestBuildExclusionZoneGeometries:
    """Tests that _build_exclusion_zone_geometries computes zones from config."""

    def _make_coord_system(self):
        cs = MagicMock()
        cs.latlon_to_scene.return_value = (50.0, 60.0)
        return cs

    def _make_config(self, user_assets=None, xml_scenes=None):
        cfg = MagicMock()
        cfg.vegetation_exclusion_zones = []
        cfg.user_assets = user_assets or []
        cfg.xml_scenes = xml_scenes or []
        return cfg

    def _make_user_asset(
        self, object_id, exclusion_zone, coordinate=(5.0, 10.0), coord_type="scene"
    ):
        from s2gos_utils.io.paths import PathRef

        return UserAssets.model_construct(
            object_id=object_id,
            exclusion_zone=exclusion_zone,
            coordinate=list(coordinate),
            coord_type=coord_type,
            ply_path=PathRef("/fake/obj.ply"),
            material="diffuse",
            elevation_offset=0.0,
            scale=1.0,
            rotation_x=0.0,
            rotation_y=0.0,
            rotation_z=0.0,
            blender_fix=False,
            face_normals=None,
        )

    def _make_xml_scene(
        self, exclusion_zone, base_coordinate=(3.0, 7.0), coord_type="scene"
    ):
        from s2gos_utils.io.paths import PathRef

        return XmlSceneConfig.model_construct(
            xml_path=PathRef("/fake/scene.xml"),
            base_coordinate=base_coordinate,
            coord_type=coord_type,
            exclusion_zone=exclusion_zone,
            object_id_prefix=None,
            elevation_offset=0.0,
            scale=1.0,
            fix_blender_coords=True,
            rotation_x=0.0,
            rotation_y=0.0,
            rotation_z=0.0,
            material_mappings=[],
            validate_materials=True,
        )

    def test_user_asset_circular_zone_built(self):
        from shapely.geometry import Point

        from s2gos_generator.core.context import _build_exclusion_zone_geometries

        asset = self._make_user_asset(
            "tree", exclusion_zone=5.0, coordinate=(10.0, 20.0), coord_type="scene"
        )
        config = self._make_config(user_assets=[asset])
        cs = self._make_coord_system()

        result = _build_exclusion_zone_geometries(config, cs)

        assert len(result) == 1
        assert result[0]["source"] == "asset_tree"
        assert result[0]["geometry"].contains(Point(10.0, 20.0))

    def test_user_asset_box_zone_built(self):
        from s2gos_generator.core.context import _build_exclusion_zone_geometries

        asset = self._make_user_asset(
            "building",
            exclusion_zone=(10.0, 6.0),
            coordinate=(0.0, 0.0),
            coord_type="scene",
        )
        config = self._make_config(user_assets=[asset])
        cs = self._make_coord_system()

        result = _build_exclusion_zone_geometries(config, cs)

        assert len(result) == 1
        assert result[0]["source"] == "asset_building"
        assert result[0]["geometry"].bounds == pytest.approx((-5.0, -3.0, 5.0, 3.0))

    def test_none_exclusion_zone_skipped(self):
        from s2gos_generator.core.context import _build_exclusion_zone_geometries

        asset = self._make_user_asset("tree", exclusion_zone=None)
        config = self._make_config(user_assets=[asset])
        cs = self._make_coord_system()

        result = _build_exclusion_zone_geometries(config, cs)
        assert result == []

    def test_xml_scene_circular_zone_built(self):
        from shapely.geometry import Point

        from s2gos_generator.core.context import _build_exclusion_zone_geometries

        xml_scene = self._make_xml_scene(
            exclusion_zone=4.0, base_coordinate=(3.0, 7.0), coord_type="scene"
        )
        config = self._make_config(xml_scenes=[xml_scene])
        cs = self._make_coord_system()

        result = _build_exclusion_zone_geometries(config, cs)

        assert len(result) == 1
        assert result[0]["source"].startswith("xml_scene_")
        assert result[0]["geometry"].contains(Point(3.0, 7.0))

    def test_combined_asset_and_xml_scene_zones(self):
        from s2gos_generator.core.context import _build_exclusion_zone_geometries

        asset = self._make_user_asset(
            "tree", exclusion_zone=5.0, coordinate=(10.0, 0.0), coord_type="scene"
        )
        xml_scene = self._make_xml_scene(
            exclusion_zone=3.0, base_coordinate=(0.0, 10.0), coord_type="scene"
        )
        config = self._make_config(user_assets=[asset], xml_scenes=[xml_scene])
        cs = self._make_coord_system()

        result = _build_exclusion_zone_geometries(config, cs)

        assert len(result) == 2
        sources = {r["source"] for r in result}
        assert "asset_tree" in sources
        assert any(s.startswith("xml_scene_") for s in sources)


class TestProcessUserAssets:
    """Tests for process_user_assets actual behavior using real file I/O."""

    def _make_ctx(self, tmp_path, user_assets=None):
        ctx = MagicMock()
        ctx.output_dir = tmp_path / "output"
        ctx.data_dir = tmp_path / "data"
        ctx.user_assets = user_assets or []
        ctx.dependency_outputs = {"target_dem": tmp_path / "dem.tif"}
        cs = MagicMock()
        cs.scene_to_latlon.return_value = (45.0, 10.0)
        cs.query_height_from_dem.return_value = 100.0
        ctx.coordinate_system = cs
        ctx.assets = MagicMock()
        ctx.assets.user_assets_file = None
        return ctx

    def _make_asset(self, tmp_path, object_id="tree", material="diffuse"):
        ply = tmp_path / "mesh.ply"
        ply.write_bytes(b"")
        return UserAssets.model_construct(
            object_id=object_id,
            ply_path=ply,
            coordinate=[5.0, 10.0],
            coord_type="scene",
            material=material,
            elevation_offset=0.0,
            scale=1.0,
            rotation_x=0.0,
            rotation_y=0.0,
            rotation_z=0.0,
            blender_fix=False,
            face_normals=None,
            exclusion_zone=None,
        )

    def _io_patches(self):
        return (
            patch(_PATCH_MKDIR, lambda p: Path(p).mkdir(parents=True, exist_ok=True)),
            patch(_PATCH_COPY, shutil.copy2),
            patch(_PATCH_OPEN_FILE, open),
        )

    def test_no_sidecar_when_no_assets(self, tmp_path):
        from s2gos_generator.resources.assets import process_user_assets

        ctx = self._make_ctx(tmp_path, user_assets=[])
        with patch(_PATCH_MKDIR, lambda p: Path(p).mkdir(parents=True, exist_ok=True)):
            result = process_user_assets(ctx)

        assert not (tmp_path / "data" / "user_assets.yml").exists()
        assert result is None

    def test_sidecar_written_with_correct_structure(self, tmp_path):
        from s2gos_generator.resources.assets import process_user_assets

        asset = self._make_asset(tmp_path, object_id="building", material="concrete")
        ctx = self._make_ctx(tmp_path, user_assets=[asset])

        mkdir_patch, copy_patch, open_patch = self._io_patches()
        with mkdir_patch, copy_patch, open_patch:
            process_user_assets(ctx)

        sidecar = tmp_path / "data" / "user_assets.yml"
        assert sidecar.exists()
        data = yaml.safe_load(sidecar.read_text())
        assert "objects" in data
        assert len(data["objects"]) == 1
        obj = data["objects"][0]
        assert obj["id"] == "building"
        assert "mesh" in obj
        assert "position" in obj
        assert "materials" not in data

    def test_inline_material_extracted_to_sidecar(self, tmp_path):
        from s2gos_generator.resources.assets import process_user_assets

        mat_dict = {"type": "diffuse", "reflectance": 0.4}
        asset = self._make_asset(tmp_path, object_id="rock", material=mat_dict)
        ctx = self._make_ctx(tmp_path, user_assets=[asset])

        mkdir_patch, copy_patch, open_patch = self._io_patches()
        with mkdir_patch, copy_patch, open_patch:
            process_user_assets(ctx)

        sidecar = tmp_path / "data" / "user_assets.yml"
        data = yaml.safe_load(sidecar.read_text())
        assert "objects" in data
        assert "materials" in data
        assert "rock_material" in data["materials"]
        assert data["objects"][0]["material"] == "rock_material"

    def test_ply_copied_to_objects_dir(self, tmp_path):
        from s2gos_generator.resources.assets import process_user_assets

        asset = self._make_asset(tmp_path, object_id="cactus")
        ctx = self._make_ctx(tmp_path, user_assets=[asset])

        mkdir_patch, copy_patch, open_patch = self._io_patches()
        with mkdir_patch, copy_patch, open_patch:
            process_user_assets(ctx)

        assert (tmp_path / "output" / "objects" / "cactus.ply").exists()
