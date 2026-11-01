"""Kayan pencere sayacı testleri (sahte saatle, bekleme yok)."""

import pytest

from app.services.rate_limiter import Rate, SlidingWindowLimiter, parse_rate


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("20/minute", Rate(20, 60)),
        ("5/second", Rate(5, 1)),
        ("100/hour", Rate(100, 3600)),
        ("1/day", Rate(1, 86400)),
        ("30/10seconds", Rate(30, 10)),
        (" 10 / Minutes ", Rate(10, 60)),
        ("3/min", Rate(3, 60)),
    ],
)
def test_parse_rate(spec, expected):
    assert parse_rate(spec) == expected


@pytest.mark.parametrize("spec", ["", "abc", "20", "20/week", "0/minute", "-1/minute", "/minute"])
def test_parse_rate_invalid(spec):
    with pytest.raises(ValueError):
        parse_rate(spec)


def test_allows_up_to_limit_then_blocks_with_retry_after():
    clock = Clock()
    limiter = SlidingWindowLimiter(clock=clock)
    rate = parse_rate("3/minute")

    results = [limiter.hit("u1", rate) for _ in range(3)]
    assert [r.allowed for r in results] == [True, True, True]
    assert [r.remaining for r in results] == [2, 1, 0]

    clock.now += 10
    blocked = limiter.hit("u1", rate)
    assert not blocked.allowed and blocked.remaining == 0
    assert blocked.retry_after == 50  # ilk istek 60 sn sonra düşer, 10 sn geçti


def test_window_slides_and_blocked_hits_are_not_counted():
    clock = Clock()
    limiter = SlidingWindowLimiter(clock=clock)
    rate = Rate(2, 10)
    assert limiter.hit("k", rate).allowed  # t=0
    clock.now += 5
    assert limiter.hit("k", rate).allowed  # t=5
    for _ in range(20):  # engellenen denemeler sayaç şişirmemeli
        assert not limiter.hit("k", rate).allowed
    clock.now += 5.1  # t=10.1: ilk istek pencereden düştü
    assert limiter.hit("k", rate).allowed
    assert not limiter.hit("k", rate).allowed  # t=5 ve t=10.1 hâlâ pencerede


def test_keys_are_independent():
    limiter = SlidingWindowLimiter(clock=Clock())
    rate = Rate(1, 60)
    assert limiter.hit("chat:1", rate).allowed
    assert not limiter.hit("chat:1", rate).allowed
    assert limiter.hit("chat:2", rate).allowed
    assert limiter.hit("upload:1", rate).allowed


def test_prune_bounds_memory():
    clock = Clock()
    limiter = SlidingWindowLimiter(clock=clock, max_keys=5)
    rate = Rate(1, 10)
    for i in range(5):
        limiter.hit(f"k{i}", rate)
    clock.now += 11  # hepsi süresi dolmuş
    limiter.hit("new", rate)
    assert len(limiter._hits) == 1
