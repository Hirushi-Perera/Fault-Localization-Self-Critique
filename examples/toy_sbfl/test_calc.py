from calc import add, subtract, percentage


def test_add():
    assert add(2, 3) == 5            # passes


def test_subtract():
    assert subtract(5, 3) == 2       # passes


def test_percentage():
    assert percentage(1, 4) == 25.0  # FAILS - percentage() multiplies by 10, not 100
