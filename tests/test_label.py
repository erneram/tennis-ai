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
