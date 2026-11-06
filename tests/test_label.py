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

    label.save_agreement("v1.mp4", "1", NONE, " both saw it up ")
    rows = label.read(tmp_path / "labels.csv")
    assert [(r["video"], r["category"], r["drill_id"]) for r in rows] == [
        ("v0.mp4", DROP, "toss_hold"), ("v1.mp4", NONE, "steady_serves"), ("v2.mp4", "", "")
    ]  # fmt: skip
    assert rows[1]["notes"] == "both saw it up"
    assert label.agree_view()[1].startswith("### Listo")
