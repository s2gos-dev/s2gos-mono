import pathlib
from pathlib import Path

import pytest

from s2gos_utils.io import FileResolver


def test_resolver():
    resolver = FileResolver()
    assert resolver.paths == []

    filepath = pathlib.Path(__file__).parent.resolve()
    resolver = FileResolver(paths=[filepath])
    assert resolver.paths == [filepath]


def test_append():
    filepath = pathlib.Path(__file__).parent.resolve()
    resolver = FileResolver(paths=[filepath])

    resolver.append(filepath, avoid_duplicates=True)
    assert resolver.paths == [filepath]

    resolver.append(filepath / "..", avoid_duplicates=True)
    assert resolver.paths == [filepath, filepath / ".."]

    resolver.append(filepath, avoid_duplicates=False)
    assert resolver.paths == [filepath, filepath / "..", filepath]

    with pytest.raises(FileNotFoundError):
        resolver.append(Path("/some/path/that/does/not/exist"), avoid_duplicates=False)


def test_prepend():
    filepath = pathlib.Path(__file__).parent.resolve()
    resolver = FileResolver(paths=[filepath])

    resolver.prepend(filepath, avoid_duplicates=True)
    assert resolver.paths == [filepath]

    resolver.prepend(filepath / "..", avoid_duplicates=True)
    assert resolver.paths == [filepath / "..", filepath]

    resolver.prepend(filepath, avoid_duplicates=False)
    assert resolver.paths == [filepath, filepath / "..", filepath]

    with pytest.raises(FileNotFoundError):
        resolver.prepend(Path("/some/path/that/does/not/exist"), avoid_duplicates=False)


def test_clear():
    filepath = pathlib.Path(__file__).parent.resolve()
    resolver = FileResolver(paths=[filepath])
    resolver.clear()

    assert resolver.paths == []


def test_resolve(tmp_path):
    tmp_sub_path = tmp_path / "sub"
    tmp_sub_path.mkdir()

    resolver = FileResolver(paths=[tmp_path])
    resolved_path = resolver.resolve(Path("sub"), strict=True)
    assert resolved_path == tmp_sub_path

    resolved_path = resolver.resolve(tmp_sub_path, strict=True)
    assert resolved_path == tmp_sub_path

    with pytest.raises(FileNotFoundError):
        resolved_path = resolver.resolve(Path("not_found"), strict=True)
