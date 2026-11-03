from sprout.serve.types import STATUSES, Candidate, FakeIdentifier, Identification


def _all() -> list[Identification]:
    top = Candidate("tomato_early_blight", 0.9)
    alt = [Candidate("tomato_late_blight", 0.05)]
    return [
        Identification("answered", top, alt, "see source"),
        Identification("unsure", None, alt, "retake the photo"),
        Identification("not_plant", None, [], "not a plant"),
        Identification("invalid", None, [], "unreadable image"),
    ]


def test_fake_returns_each_status():
    fake = FakeIdentifier(_all())
    got = [fake.identify(b"x").status for _ in range(4)]
    assert got == list(STATUSES)


def test_identification_round_trips_as_json():
    for r in _all():
        assert Identification.from_json(r.to_json()) == r
