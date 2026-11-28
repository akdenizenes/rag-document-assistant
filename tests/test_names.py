import pytest

from app.services.names import clean_full_name


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Enes Akdeniz", "Enes Akdeniz"),
        ("  Enes    Akdeniz ", "Enes Akdeniz"),
        ("Çağrı Şükrü Öztürk", "Çağrı Şükrü Öztürk"),
        ("Anne-Marie O'Neil", "Anne-Marie O'Neil"),
        ("M. Ali", "M. Ali"),
        ("Su", "Su"),
    ],
)
def test_valid_names(raw, expected):
    assert clean_full_name(raw) == expected


@pytest.mark.parametrize("raw", ["", "   ", "a", "123", "!!!", "Enes123", "-Enes", "<b>Enes</b>", "**Enes**"])
def test_invalid_names(raw):
    with pytest.raises(ValueError):
        clean_full_name(raw)
