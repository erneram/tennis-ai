import pytest

from eval import label

DROP, NONE = "toss_arm_drops_early", "no_clear_problem"


def test_save_agreement_writes_agreed_rows_and_decisions(tmp_path, monkeypatch):
    monkeypatch.setattr(label, "EVAL", tmp_path)
    monkeypatch.setattr(label, "NAMES", ["a", "b"])
    monkeypatch.setattr(label, "AGREED", tmp_path / "labels.csv")
    monkeypatch.setattr(label, "playable", lambda name, slowdown: name)
    answers = {"a": [("1", DROP, "low"), ("1", NONE, ""), ("0", "", "")],
               "b": [("1", DROP, ""), ("1", DROP, ""), ("0", "", "")]}  # fmt: skip
    for name, rows in answers.items():
        rows = [label.answered({"video": f"v{i}.mp4", "player": "jc"}, v, c, n)
                for i, (v, c, n) in enumerate(rows)]  # fmt: skip
        label.write(tmp_path / f"labels_{name}.csv", rows, label.LABEL_COLUMNS)

    agreed, differ = label.disputes()
    assert [r["video"] for r in agreed] == ["v0.mp4", "v2.mp4"]
    assert agreed[0]["notes"] == "a: low"
    assert list(differ) == ["v1.mp4"]
    assert label.agree_view()[1] == "### Decididos 0 de 1"

    other = label.answered({"video": "pro09-s01.mp4", "player": "pro09"}, "1", NONE, "claude")
    label.write(tmp_path / "labels.csv", [other], label.LABEL_COLUMNS)  # not a member's label
    label.save_agreement("v1.mp4", "1", NONE, " both saw it up ")
    rows = label.read(tmp_path / "labels.csv")
    assert [(r["video"], r["category"], r["drill_id"]) for r in rows] == [
        ("v0.mp4", DROP, "toss_hold"), ("v1.mp4", NONE, "steady_serves"), ("v2.mp4", "", ""),
        ("pro09-s01.mp4", NONE, "steady_serves"),
    ]  # fmt: skip
    assert rows[1]["notes"] == "both saw it up"
    assert label.agree_view()[1].startswith("### Listo")


def test_new_clips_in_raw_are_listed_until_labeled(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    (raw / "uncut").mkdir(parents=True)
    for name in ("old.mp4", "ea-2-s01.mp4", "pro09-s01.mp4", "uncut/session.MOV", "cuts.csv"):
        (raw / name).touch()
    monkeypatch.setattr(label, "EVAL", tmp_path)
    monkeypatch.setattr(label, "AGREED", tmp_path / "labels.csv")
    monkeypatch.setattr(label, "RAW", raw)
    other = label.answered({"video": "pro09-s01.mp4", "player": "pro09"}, "1", NONE, "claude")
    label.write(tmp_path / "labels.csv", [other], label.LABEL_COLUMNS)  # at the team's request
    monkeypatch.setattr(label, "playable", lambda name, slowdown: name)
    old = label.answered({"video": "old.mp4", "player": "jc"}, "0", None, "")
    label.write(tmp_path / "labels_a.csv", [old], label.LABEL_COLUMNS)

    choices, status = label.label_view("a")[:2]
    assert choices["choices"] == ["ea-2-s01.mp4"] and status.startswith("Faltan 1 de 2")
    label.save_label("a", "ea-2-s01.mp4", "1", NONE, "")
    rows = label.read(tmp_path / "labels_a.csv")
    assert [(r["video"], r["player"], r["category"]) for r in rows] == [
        ("old.mp4", "jc", ""), ("ea-2-s01.mp4", "ea", NONE)
    ]  # fmt: skip


def test_body_answers_take_players_in_turn_and_need_every_question(tmp_path, monkeypatch):
    monkeypatch.setattr(label, "EVAL", tmp_path)
    monkeypatch.setattr(label, "AGREED", tmp_path / "labels.csv")
    monkeypatch.setattr(label, "playable", lambda name, slowdown: name)
    videos = [("jc-s1", "jc", "1"), ("jc-s2", "jc", "1"), ("jc-s3", "jc", "0"),
              ("jc-post1-s01", "jc", "1"), ("pro01_s01", "pro01", "1")]  # fmt: skip
    rows = [label.answered({"video": f"{v}.mp4", "player": p}, valid, NONE, "")
            for v, p, valid in videos]  # fmt: skip
    label.write(tmp_path / "labels.csv", rows, label.LABEL_COLUMNS)

    choices, status = label.aspect_view("a")[:2]
    assert choices["choices"] == ["jc-s1.mp4", "jc-post1-s01.mp4", "pro01_s01.mp4", "jc-s2.mp4"]
    assert status.startswith("Faltan 4 de 4")
    answers = [options[0][1] for *_, options in label.ASPECTS]
    with pytest.raises(label.gr.Error):
        label.save_aspects("a", "jc-s1.mp4", *answers[:-1], None, "")
    out = label.save_aspects("a", "jc-s1.mp4", *answers, " low toss ")
    assert out[0]["value"] == "jc-post1-s01.mp4" and out[1].startswith("Faltan 3 de 4")
    [saved] = label.read(tmp_path / "aspects_a.csv")
    assert (
        saved["video"] == "jc-s1.mp4" and saved["legs"] == "none" and saved["notes"] == "low toss"
    )
    assert label.aspect_view("a", "jc-s1.mp4")[4] == "none"  # the saved answer comes back


def test_one_rater_clips_count_alone_and_merge_keeps_the_decisions(tmp_path, monkeypatch):
    monkeypatch.setattr(label, "EVAL", tmp_path)
    monkeypatch.setattr(label, "NAMES", ["a", "b"])
    monkeypatch.setattr(label, "AGREED", tmp_path / "labels.csv")
    a = [("v0.mp4", DROP, ""), ("v1.mp4", NONE, ""), ("yt9-s01.mp4", NONE, "side"),
         ("v2.mp4", NONE, "")]  # fmt: skip
    b = [("v0.mp4", DROP, ""), ("v1.mp4", DROP, "")]
    for name, rows in (("a", a), ("b", b)):
        rows = [label.answered({"video": v, "player": "jc"}, "1", c, n) for v, c, n in rows]
        label.write(tmp_path / f"labels_{name}.csv", rows, label.LABEL_COLUMNS)

    agreed, differ = label.disputes()
    assert [r["video"] for r in agreed] == ["v0.mp4", "yt9-s01.mp4"]  # v2.mp4 waits for b
    assert agreed[1]["notes"] == "one rater: a; a: side"
    assert list(differ) == ["v1.mp4"]
    decision = label.answered(differ["v1.mp4"][0], "1", NONE, "settled")
    label.write(tmp_path / "labels.csv", [decision], label.LABEL_COLUMNS)
    label.merge()
    rows = label.read(tmp_path / "labels.csv")
    assert [(r["video"], r["category"], r["notes"]) for r in rows] == [
        ("v0.mp4", DROP, ""), ("v1.mp4", NONE, "settled"),
        ("yt9-s01.mp4", NONE, "one rater: a; a: side"),
    ]  # fmt: skip


def test_retest_draws_answered_serves_once_and_hides_their_names(tmp_path, monkeypatch):
    monkeypatch.setattr(label, "EVAL", tmp_path)
    monkeypatch.setattr(label, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(label, "RETEST", 3)
    monkeypatch.setattr(label, "playable", lambda name, slowdown: str(_clip(tmp_path, name)))
    answers = [options[0][1] for *_, options in label.ASPECTS]
    first = [{"video": f"jc-s{i}.mp4", "player": "jc",
              **dict(zip((a[0] for a in label.ASPECTS), answers if i < 5 else [""] * 7)),
              "notes": ""} for i in range(7)]  # fmt: skip
    label.write(tmp_path / "aspects_a.csv", first, label.ASPECT_COLUMNS)
    assert label.retest_view("b")[1] == "Primero contesta «Cuerpo»."

    drawn = [r["video"] for r in label.retest_rows("a")]
    assert len(drawn) == 3 and set(drawn) <= {f"jc-s{i}.mp4" for i in range(5)}
    (tmp_path / "aspects_a_retest.csv").unlink()
    assert [r["video"] for r in label.retest_rows("a")] == drawn  # same seed, same draw
    choices, status, normal, slow, *rest = label.retest_view("a")
    assert [c[0] for c in choices["choices"]] == ["Saque 1", "Saque 2", "Saque 3"]
    assert status == "Faltan 3 de 3 saques." and rest[:-1] == [None] * 7  # no first answers
    assert "jc-s" not in normal and "jc-s" not in slow
    out = label.save_retest("a", "0", *answers, "")
    assert out[0]["value"] == "1" and out[1] == "Faltan 2 de 3 saques."
    saved = label.read(tmp_path / "aspects_a_retest.csv")
    assert [r["video"] for r in saved] == drawn and saved[0]["legs"] == answers[0]


def _clip(folder, name):
    path = folder / name
    path.touch()
    return path
