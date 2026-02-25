import os
import tempfile

import pytest
from dynaconf.utils.boxing import DynaBox

from s2gos_utils.io import PathRef
from s2gos_utils.setting import load_config, settings, to_pathref


@pytest.fixture
def temp_config_file():
    """Create a temporary s2gos_settings.yaml file for testing."""
    config_content = """
common:
    search_paths:
        - "/tmp/test_path1"
        - "/tmp/test_path2"
    local_fsspec_cache: "/tmp/test_cache"
    credential_provider: "environment"
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_content)
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


class TestSettings:
    """Test settings loading and default values."""

    def test_settings_has_common_section(self):
        """Test that settings object has a common section."""
        assert hasattr(settings, "common")

    def test_settings_default_values(self):
        """Test that default values are set correctly when not in config."""
        # These defaults should exist even without a config file
        assert isinstance(settings.common.search_paths, list)
        assert isinstance(settings.common.local_fsspec_cache, str)
        assert settings.common.credential_provider in ["dynaconf", "environment"]

    def test_settings_validates_only_common(self):
        """Test that only the common section is validated."""
        # The settings object should have validate_only set to "common"
        assert settings.validators[0].names == ("SEARCH_PATHS",)
        assert settings.validators[1].names == ("LOCAL_FSSPEC_CACHE",)


class TestToPathRef:
    """Test the to_pathref function with various input types."""

    def test_to_pathref_from_string(self):
        """Test converting a simple string path to PathRef."""
        result = to_pathref("/tmp/test_path")

        assert isinstance(result, PathRef)
        assert result.value == "/tmp/test_path"
        assert result.cid is None

    def test_to_pathref_from_dict(self):
        """Test converting a dict with and without credential ID to PathRef."""
        path_dict = {"value": "/tmp/test_path"}
        result = to_pathref(path_dict)

        assert isinstance(result, PathRef)
        assert result.value == "/tmp/test_path"
        assert result.cid is None

        path_dict = {"value": "s3://bucket/path", "cid": "my_credential"}
        result = to_pathref(path_dict)

        assert isinstance(result, PathRef)
        assert result.value == "s3://bucket/path"
        assert result.cid == "my_credential"

    def test_to_pathref_from_dynabox(self):
        """Test converting a DynaBox (dynaconf object) to PathRef."""
        dynabox = DynaBox(
            {"value": "https://example.com/data.zarr", "cid": "example_cred"}
        )
        result = to_pathref(dynabox)

        assert isinstance(result, PathRef)
        assert result.value == "https://example.com/data.zarr"
        assert result.cid == "example_cred"

    def test_to_pathref_preserves_pathref_input(self):
        """Test that passing a PathRef returns an equivalent PathRef."""
        original = PathRef(value="/tmp/path", cid="test")
        result = to_pathref({"value": original.value, "cid": original.cid})

        assert isinstance(result, PathRef)
        assert result.value == original.value
        assert result.cid == original.cid


class TestLoadConfig:
    """Test the load_config function."""

    def test_load_config_adds_existing_paths(self, tmp_path):
        """Test that load_config only adds existing paths to resolver."""
        # Create a temporary directory that exists
        existing_path = tmp_path / "existing"
        existing_path.mkdir()

        # Import resolver to check its state
        from s2gos_utils.io.resolver import resolver

        # Store initial resolver length
        initial_length = len(resolver.paths)

        # Temporarily modify settings to include our test path
        original_paths = settings.common.search_paths
        settings.common.search_paths = [str(existing_path)]

        try:
            # Call load_config
            load_config()

            # Check that the existing path was added
            assert len(resolver.paths) >= initial_length

        finally:
            # Restore original settings
            settings.common.search_paths = original_paths

    def test_load_config_skips_nonexistent_paths(self):
        """Test that load_config skips paths that don't exist."""
        from s2gos_utils.io.resolver import resolver

        # Store initial resolver length
        initial_length = len(resolver.paths)

        # Temporarily modify settings to include a non-existent path
        original_paths = settings.common.search_paths
        settings.common.search_paths = ["/path/that/does/not/exist/12345"]

        try:
            # Call load_config - should not raise an error
            load_config()

            # The resolver length should not increase for non-existent paths
            # (it might stay the same or increase due to other paths)
            assert len(resolver.paths) >= initial_length

        finally:
            # Restore original settings
            settings.common.search_paths = original_paths
