from datetime import date

import pytest

from s2gos_simulator.acquisitions._fetch import (
    FootprintOutsideProduct,
    closest_fitting,
)

_ANCHOR = date(2024, 6, 11)


def _name(day: int, time: str = "083306", timeliness: str = "NT") -> str:
    return (
        f"S3B_OL_1_EFR____202406{day:02d}T{time}_202406{day:02d}T{time}_"
        f"20240921T163355_0179_094_078_3420_MAR_O_{timeliness}_004.SEN3"
    )


def _search(cache, names, downloaded=None):
    """A catalog offering `names`, recording what gets downloaded."""

    def download(name):
        if downloaded is not None:
            downloaded.append(name)
        path = cache / name
        path.mkdir(parents=True)
        return path

    return lambda: {name: (lambda name=name: download(name)) for name in names}


def _cached(cache, *names):
    for name in names:
        (cache / name).mkdir(parents=True)


def _rejecting(*names):
    def check(path):
        if path.name in names:
            raise FootprintOutsideProduct("does not fit")

    return check


def test_products_rank_by_day_then_timeliness_then_start(tmp_path):
    tried = []

    def reject_all(path):
        tried.append(path.name)
        raise FootprintOutsideProduct("does not fit")

    ranked = [
        _name(11, "090000"),
        _name(11, "100000"),
        _name(11, "080000", "NR"),
        _name(10),
        _name(13),
    ]

    with pytest.raises(RuntimeError, match="holds the footprint"):
        closest_fitting(
            tmp_path, _ANCHOR, reject_all, _search(tmp_path, reversed(ranked))
        )

    assert tried == ranked


def test_rejected_product_moves_on_and_warns(tmp_path, caplog):
    closest, next_day = _name(11), _name(12)

    with caplog.at_level("WARNING"):
        path = closest_fitting(
            tmp_path,
            _ANCHOR,
            _rejecting(closest),
            _search(tmp_path, [closest, next_day]),
        )

    assert path.name == next_day
    assert "using the 2024-06-12 pass" in caplog.text
    assert closest in caplog.text


def test_closest_product_on_the_day_gives_no_warning(tmp_path, caplog):
    with caplog.at_level("WARNING"):
        path = closest_fitting(
            tmp_path, _ANCHOR, _rejecting(), _search(tmp_path, [_name(11)])
        )

    assert path.name == _name(11)
    assert not caplog.text


def test_cached_product_is_used_without_searching(tmp_path):
    _cached(tmp_path, _name(11))

    def search():
        raise AssertionError("searched the catalogue")

    assert closest_fitting(tmp_path, _ANCHOR, _rejecting(), search).name == _name(11)


def test_cached_product_too_small_moves_on_without_downloading_it_again(tmp_path):
    _cached(tmp_path, _name(11))
    downloaded = []

    path = closest_fitting(
        tmp_path,
        _ANCHOR,
        _rejecting(_name(11)),
        _search(tmp_path, [_name(11), _name(12)], downloaded),
    )

    assert path.name == _name(12)
    assert downloaded == [_name(12)]


def test_other_errors_stop_the_search(tmp_path):
    downloaded = []

    def check(path):
        raise ValueError("unknown band")

    with pytest.raises(ValueError, match="unknown band"):
        closest_fitting(
            tmp_path,
            _ANCHOR,
            check,
            _search(tmp_path, [_name(11), _name(12)], downloaded),
        )

    assert downloaded == [_name(11)]


def test_partial_downloads_are_ignored(tmp_path):
    _cached(tmp_path, f"{_name(11)}.part")

    path = closest_fitting(
        tmp_path, _ANCHOR, _rejecting(), _search(tmp_path, [_name(12)])
    )

    assert path.name == _name(12)
