from strokes import base, serve


def test_every_category_has_a_drill_and_a_known_phase():
    criteria = base.load(serve, "criteria.yaml")
    drill_categories = {d["category"] for d in base.load(serve, "drills.yaml")["drills"]}
    assert set(criteria["categories"]) == drill_categories
    assert {c["phase"] for c in criteria["categories"].values()} <= {*serve.PHASES, "none"}
    assert list(criteria["phases"]) == list(serve.PHASES)  # same order: prompts list them in order


def test_every_phase_says_how_to_find_it_and_what_to_observe():
    for spec in base.load(serve, "criteria.yaml")["phases"].values():
        assert spec["looks_like"].strip() and spec["observe"].strip()
