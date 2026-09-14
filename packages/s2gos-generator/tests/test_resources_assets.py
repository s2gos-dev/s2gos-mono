from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError
from shapely.geometry import Point

from s2gos_generator.core.config import (
    BoxGeometry,
    CircleGeometry,
    ExclusionZone,
    ObjectExclusionZone,
    PolygonGeometry,
)
from s2gos_generator.core.config.assets import UserAssets, XmlSceneConfig
from s2gos_generator.processors.exclusion import resolve_exclusion_zones

_PATCH_RESOLVE = "s2gos_generator.core.config.assets.resolve_asset_path"
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


class TestResolveExclusionZones:
    """resolve_exclusion_zones collects config zones and object shorthand zones."""

    def _make_coord_system(self):
        cs = MagicMock()
        cs.latlon_to_scene.return_value = (50.0, 60.0)
        return cs

    def _make_config(self, exclusion_zones=None, user_assets=None, xml_scenes=None):
        cfg = MagicMock()
        cfg.exclusion_zones = exclusion_zones or []
        cfg.user_assets = user_assets or []
        cfg.xml_scenes = xml_scenes or []
        return cfg

    def _make_user_asset(
        self, object_id, exclusion_zone, coordinate=(5.0, 10.0), coord_type="scene"
    ):
        from s2gos_utils.io.paths import PathRef

        return UserAssets.model_construct(
            object_id=object_id,
            exclusion_zone=(
                None
                if exclusion_zone is None
                else ObjectExclusionZone.model_validate(exclusion_zone)
            ),
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
            exclusion_zone=(
                None
                if exclusion_zone is None
                else ObjectExclusionZone.model_validate(exclusion_zone)
            ),
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
        asset = self._make_user_asset(
            "tree", exclusion_zone=5.0, coordinate=(10.0, 20.0), coord_type="scene"
        )
        result = resolve_exclusion_zones(
            self._make_config(user_assets=[asset]), self._make_coord_system()
        )

        assert len(result) == 1
        assert result[0].source == "asset_tree"
        assert result[0].geometry.contains(Point(10.0, 20.0))
        assert result[0].excludes == {"vegetation", "buildings"}

    def test_user_asset_box_zone_built(self):
        asset = self._make_user_asset(
            "building",
            exclusion_zone=(10.0, 6.0),
            coordinate=(0.0, 0.0),
            coord_type="scene",
        )
        result = resolve_exclusion_zones(
            self._make_config(user_assets=[asset]), self._make_coord_system()
        )

        assert len(result) == 1
        assert result[0].source == "asset_building"
        assert result[0].geometry.bounds == pytest.approx((-5.0, -3.0, 5.0, 3.0))

    def test_user_asset_narrowed_excludes(self):
        asset = self._make_user_asset(
            "tower", exclusion_zone={"radius": 3.0, "excludes": ["buildings"]}
        )
        result = resolve_exclusion_zones(
            self._make_config(user_assets=[asset]), self._make_coord_system()
        )
        assert result[0].excludes == {"buildings"}

    def test_none_exclusion_zone_skipped(self):
        asset = self._make_user_asset("tree", exclusion_zone=None)
        result = resolve_exclusion_zones(
            self._make_config(user_assets=[asset]), self._make_coord_system()
        )
        assert result == []

    def test_xml_scene_circular_zone_built(self):
        xml_scene = self._make_xml_scene(
            exclusion_zone=4.0, base_coordinate=(3.0, 7.0), coord_type="scene"
        )
        result = resolve_exclusion_zones(
            self._make_config(xml_scenes=[xml_scene]), self._make_coord_system()
        )

        assert len(result) == 1
        assert result[0].source.startswith("xml_scene_")
        assert result[0].geometry.contains(Point(3.0, 7.0))

    def test_config_zones_circle_box_polygon(self):
        zones = [
            ExclusionZone(
                zone_id="c",
                geometry=CircleGeometry(
                    center=(0.0, 0.0), coord_type="scene", radius=5.0
                ),
            ),
            ExclusionZone(
                zone_id="b",
                geometry=BoxGeometry(
                    center=(10.0, 20.0), coord_type="geographic", width=4, height=2
                ),
                excludes=["vegetation"],
            ),
            ExclusionZone(
                zone_id="p",
                geometry=PolygonGeometry(
                    coordinates=[(0, 0), (10, 0), (10, 10)], coord_type="scene"
                ),
            ),
        ]
        result = resolve_exclusion_zones(
            self._make_config(exclusion_zones=zones), self._make_coord_system()
        )

        assert [z.source for z in result] == ["zone_c", "zone_b", "zone_p"]
        assert result[0].geometry.contains(Point(0, 0))
        assert result[1].geometry.bounds == (48.0, 59.0, 52.0, 61.0)
        assert result[1].excludes == {"vegetation"}
        assert result[2].geometry.contains(Point(8, 1))

    def test_combined_asset_and_xml_scene_zones(self):
        asset = self._make_user_asset(
            "tree", exclusion_zone=5.0, coordinate=(10.0, 0.0), coord_type="scene"
        )
        xml_scene = self._make_xml_scene(
            exclusion_zone=3.0, base_coordinate=(0.0, 10.0), coord_type="scene"
        )
        result = resolve_exclusion_zones(
            self._make_config(user_assets=[asset], xml_scenes=[xml_scene]),
            self._make_coord_system(),
        )

        assert len(result) == 2
        sources = {r.source for r in result}
        assert "asset_tree" in sources
        assert any(s.startswith("xml_scene_") for s in sources)

    def test_context_exclusion_zones_for_filters_by_target(self):
        from s2gos_generator.core.context import SceneResourceContext

        zones = [
            ExclusionZone(
                zone_id="all",
                geometry=CircleGeometry(center=(0, 0), coord_type="scene", radius=1),
            ),
            ExclusionZone(
                zone_id="veg",
                geometry=CircleGeometry(center=(0, 0), coord_type="scene", radius=1),
                excludes=["vegetation"],
            ),
        ]
        ctx = SceneResourceContext.__new__(SceneResourceContext)
        ctx.config = self._make_config(exclusion_zones=zones)
        ctx._exclusion_zones = None
        ctx._coord_system = self._make_coord_system()

        assert [z.source for z in ctx.exclusion_zones_for("vegetation")] == [
            "zone_all",
            "zone_veg",
        ]
        assert [z.source for z in ctx.exclusion_zones_for("buildings")] == ["zone_all"]


class TestObjectExclusionZoneShorthand:
    def test_float_is_radius(self):
        z = ObjectExclusionZone.model_validate(15.0)
        assert (z.radius, z.size) == (15.0, None)

    def test_pair_is_size(self):
        z = ObjectExclusionZone.model_validate((20, 10))
        assert (z.radius, z.size) == (None, (20.0, 10.0))

    def test_default_excludes_everything(self):
        assert ObjectExclusionZone.model_validate(1.0).excludes == [
            "vegetation",
            "buildings",
        ]

    @pytest.mark.parametrize(
        "bad",
        [
            {},
            {"radius": 1.0, "size": (1.0, 1.0)},
            {"radius": 1.0, "excludes": []},
            {"radius": 1.0, "excludes": ["roads"]},
            (1.0, 2.0, 3.0),
            (0.0, 2.0),
            -1.0,
        ],
    )
    def test_invalid_rejected(self, bad):
        with pytest.raises(ValidationError):
            ObjectExclusionZone.model_validate(bad)
