import numpy as np

from strokes.serve import body, dataset, train

J = dataset.JOINTS.index
T = dataset.BEFORE + 1 + dataset.AFTER


def serve() -> np.ndarray:
    """One window: the left arm tosses (the right wrist is highest at contact) and the player
    moves toward +x."""
    x = np.zeros((1, T, len(dataset.JOINTS), 2))
    x[:, :, [J("left_shoulder"), J("right_shoulder")], 1] = -1
    x[:, dataset.BEFORE, J("right_wrist"), 1] = -2
    x[:, :, dataset.HIPS, 0] = np.linspace(0, 1, T)[None, :, None]
    return x


def test_measures_read_the_same_for_either_tossing_arm_and_camera_side():
    x, mask = serve(), np.ones((1, T, len(dataset.JOINTS)), bool)
    mirrored = x.copy()
    train.mirror(mirrored, mask.copy(), np.array([True]))  # the right arm tosses
    flipped = x * [-1, 1]  # same arm, the court on the other side of the image
    for other in (mirrored, flipped):
        for aspect, value in body.measures(other, mask).items():
            assert np.allclose(value, body.measures(x, mask)[aspect]), aspect


def test_deeper_knee_bend_and_more_forward_landing_mean_less_to_improve():
    x, mask = serve(), np.ones((1, T, len(dataset.JOINTS)), bool)
    bent = x.copy()
    bent[:, dataset.BEFORE - 20 : dataset.BEFORE - 10, dataset.HIPS, 1] += 0.3  # hips go down
    assert body.measures(bent, mask)["legs"] < body.measures(x, mask)["legs"]
    further = x.copy()
    step = np.linspace(0, 1, T - dataset.BEFORE)[None, :, None]
    further[:, dataset.BEFORE :, dataset.HIPS, 0] += step
    assert body.measures(further, mask)["landing"] < body.measures(x, mask)["landing"]


def test_targets_are_one_for_needs_work_and_leave_out_unseen_answers():
    answered = {"v1": {"legs": "little", "shoulders": "unseen", "racket_at_peak": "trophy",
                       "racket_drop": "good", "arm_at_contact": "forward", "landing": "too_much"}}  # fmt: skip
    y = body.targets(answered, ["v1", "v2"])
    assert y.shape == (2, len(body.ANSWERS))
    assert np.array_equal(y[0], [1, np.nan, 0, 0, 1, np.nan], equal_nan=True)
    assert np.isnan(y[1]).all()


def test_session_folds_keep_each_session_on_one_side():
    videos = [f"ea-{s}-s{i:02d}" for s in range(1, 7) for i in range(1, 6)] + ["pro02_s01"]
    fold = body.session_folds(videos, seed=0)
    sessions = [body.SESSION.sub("", v) for v in videos]
    assert all(len({f for f, s2 in zip(fold, sessions) if s2 == s}) == 1 for s in set(sessions))
    assert len(set(fold)) == train.FOLDS


def test_m3_has_one_output_per_aspect_and_skips_unanswered_ones():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(6, T, len(dataset.JOINTS), 2)).astype(np.float32)
    mask = np.ones(x.shape[:3], bool)
    y = np.array([[1, np.nan], [0, np.nan], [1, 0], [0, 1], [1, 1], [0, 0]], np.float32)
    model, curve = body.fit(x, mask, y, seed=0, epochs=2)
    assert body.probability(model, x, mask).shape == (6, 2)
    assert np.isfinite(curve).all()  # the unanswered (NaN) targets never reach the loss


def full_serve() -> np.ndarray:
    """A window where every aspect has something to move: the tossing (left) wrist is highest 20
    frames in, the racket wrist drops below its shoulder then is highest at contact, the hips go
    down before the jump and move into the court."""
    t, before = np.arange(T), dataset.BEFORE
    x = serve()
    y = x[0, :, :, 1]
    y[:, J("left_shoulder")], y[:, J("right_shoulder")] = -1.2, -1.0
    y[:, J("left_wrist")] = np.interp(t, [0, 20, before, T - 1], [-0.5, -2.5, 0.5, 0.5])
    y[:, J("left_elbow")] = y[:, J("left_wrist")] / 2
    y[:, J("right_wrist")] = np.interp(
        t, [0, 20, before - 10, before, T - 1], [-0.5, -1.5, -0.5, -2, 0]
    )
    y[:, J("right_elbow")] = y[:, J("right_wrist")] / 2
    y[before - 20 : before - 10, list(dataset.HIPS)] = 0.3
    y[:, [J("left_knee"), J("right_knee")]] = 1.0
    return x


def test_each_edit_moves_only_its_part_toward_needs_work():
    from eval import body as checks

    x, mask = full_serve(), np.ones((1, T, len(dataset.JOINTS)), bool)
    upper = [J(j) for j in dataset.JOINTS[:9]] + [J("left_knee"), J("right_knee")]
    racket = [J("right_elbow"), J("right_wrist")]
    moved = {"legs": (upper, 1), "shoulders": ([J("left_shoulder"), J("right_shoulder")], 1),
             "racket_at_peak": (racket, 1), "racket_drop": (racket, 1),
             "arm_at_contact": (racket, 1), "landing": (range(len(dataset.JOINTS)), 0),
             "toss": ([J("left_elbow"), J("left_wrist")], 1)}  # fmt: skip
    before = body.measures(x, mask)
    for part, (joints, axis) in moved.items():
        edited = checks.edited(part, x)
        changed = np.argwhere((edited != x).any(1)[0])  # (joint, axis) pairs that moved
        assert len(changed) and all(j in joints and a == axis for j, a in changed), part
        if part != "toss":
            assert body.measures(edited, mask)[part] > before[part], part


def test_drill_list_keeps_the_two_furthest_allowed_problems_past_their_threshold():
    from eval import body as checks

    margin = np.array([[0.3, 0.1, 0.5, -0.2], [-0.1, -0.3, np.nan, -0.5]])
    allowed = np.array([True, True, False, True])
    said, first = checks.pick(margin, allowed)
    assert said.tolist() == [[True, True, False, False], [False, False, False, False]]
    assert first[0] == 0  # the disallowed 0.5 never comes first
    # With a rank per column (the precision of each point) the highest comes first
    assert checks.pick(margin, allowed, rank=[0.2, 0.9, 0.9, 0.7])[1][0] == 1


def test_contact_moves_to_where_the_racket_wrist_is_highest():
    x, mask = full_serve(), np.ones((1, T, len(dataset.JOINTS)), bool)
    late = x[:, np.clip(np.arange(T) + 4, 0, T - 1)]  # contact was found 4 frames late
    assert body.reach(body.oriented(late, mask)[0])[0] == dataset.BEFORE - 4
    moved, right = body.oriented(late, mask, at_reach=True)[0], body.oriented(x, mask, True)[0]
    assert body.reach(moved)[0] == dataset.BEFORE
    assert np.allclose(moved[:, 4:], right[:, 4:])
    assert np.allclose(moved[:, dataset.BEFORE, dataset.HIPS].mean(1), 0)


def test_the_key_instants_are_the_knees_most_bent_and_the_racket_wrist_highest():
    x, mask = full_serve(), np.ones((1, T, len(dataset.JOINTS)), bool)
    late = x[:, np.clip(np.arange(T) + 4, 0, T - 1)]  # contact was found 4 frames late
    at = body.instants(late, mask)
    assert at["contact"][0] == dataset.BEFORE - 4
    assert at["bend"][0] == dataset.BEFORE - 20 - 4  # the first frame with the hips down


def test_the_audited_racket_drop_is_how_far_the_racket_wrist_falls():
    x, mask = full_serve(), np.ones((1, T, len(dataset.JOINTS)), bool)
    assert np.isclose(body.measures(x, mask, audited=True)["racket_drop"], -1)  # from 1.5 to 0.5
    rising = x.copy()  # the racket wrist only goes up from the trophy to contact: no drop
    rising[0, :, J("right_wrist"), 1] = np.interp(np.arange(T), [0, dataset.BEFORE], [-0.5, -2])
    assert np.isclose(body.measures(rising, mask, audited=True)["racket_drop"], 0)
    assert np.isnan(body.measures(x, mask, audited=True)["racket_at_peak"]).all()


def test_every_player_and_answer_cell_weighs_the_same():
    player = np.array(["jc"] * 12 + ["ea"] * 12 + ["pro02"] * 3 + ["pro03"] * 3)
    y = np.array([0] * 6 + [1] * 6 + [1] * 10 + [0] * 2 + [0] * 5 + [np.nan], np.float32)[:, None]
    weight = body.cell_weights(y, player)[:, 0]
    cells = [weight[:6], weight[6:12], weight[12:22], weight[24:29]]
    assert np.allclose([c.sum() for c in cells], 1)  # the professionals are one player
    assert np.allclose(weight[22:24], 1 / body.MIN_CELL)  # two serves do not stand for a cell
    assert weight[-1] == 0  # no answer


def test_m4_shows_each_aspect_only_its_part_of_the_body():
    rng = np.random.default_rng(0)
    o = rng.normal(size=(8, T, len(dataset.JOINTS), 2)).astype(np.float32)
    aspects = ["legs", "racket_at_peak", "racket_drop", "arm_at_contact", "landing"]
    spec = {"aspects": aspects, "travel": True, "parts": True, "balanced": True}
    y = rng.integers(0, 2, (8, len(aspects))).astype(np.float32)
    inputs = body.prepared(o, travel=True, parts=True)
    assert inputs["lower"][0].shape == (8, T, 10, 2) and inputs["racket"][0].shape == (8, T, 4, 1)
    networks, curve = body.fit_heads(o, y, 0, 2, spec, np.array(["jc"] * 4 + ["ea"] * 4))
    said = body.probabilities(networks, o, spec)
    assert said.shape == (8, len(aspects)) and np.isfinite(curve).all()
    arms, legs = o.copy(), o.copy()
    arms[:, :, train.ARMS] += 1
    legs[:, :, body.LOWER] += 1
    assert np.allclose(body.probabilities(networks, arms, spec)[:, [0, 4]], said[:, [0, 4]])
    assert np.allclose(body.probabilities(networks, legs, spec)[:, 1:4], said[:, 1:4])


def test_an_aspect_is_validated_on_trial_exploratory_or_neither():
    from eval import body as checks

    def table(within_ea=0.75, **changed):
        passes = {"0 retest kappa": "yes", "1 within players": "yes", "2 new players": "yes",
                  "3 professionals called needs work": "yes", "4 counterfactual": "yes",
                  "9 precision among amateurs": "yes", "1 within EA": "", "1 within JC": ""}  # fmt: skip
        value = {"1 within EA": within_ea, "1 within JC": 0.6}
        return [{"aspect": "landing", "check": check, "passes": changed.get(check, said),
                 "value": value.get(check, 1)} for check, said in passes.items()]  # fmt: skip

    assert checks.level(table(), "landing") == "validated"
    # Under its measure within a player, but it ranks EA's serves: on trial
    assert checks.level(table(**{"1 within players": "no"}), "landing") == "trial"
    # It does not carry over to a player left out, the one line it fails: exploratory
    unseen = table(**{"1 within players": "no", "2 new players": "no"})
    assert checks.level(unseen, "landing") == "exploratory"
    wrong = table(**{"1 within players": "no", "9 precision among amateurs": "no"})
    assert checks.level(wrong, "landing") == ""
    # It ranks the serves of neither player: it may only tell the players apart
    assert checks.level(table(0.6, **{"1 within players": "no"}), "landing") == ""
