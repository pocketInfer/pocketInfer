import pytest

from pocketinfer.sizes import parse_size


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("32GiB", 32 * 2**30),
        ("6GB", 6 * 10**9),
        ("1024MiB", 2**30),
        ("4096", 4096),
    ],
)
def test_parse_size(value: str, expected: int) -> None:
    assert parse_size(value) == expected


def test_parse_size_rejects_unknown_suffix() -> None:
    with pytest.raises(ValueError, match="invalid size"):
        parse_size("32 elephants")
