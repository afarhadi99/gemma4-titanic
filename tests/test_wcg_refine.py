from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from titanic.submission import validate_submission
from titanic.wcg import (
    CLASS3,
    DEFAULT,
    DIE,
    LIVE,
    Config,
    apply_rules,
    build_groups,
    evaluate_grid,
    grid_configs,
    pool_passengers,
    predict_config,
    segment_flips,
)
from wcg import run_grid

FIXTURES = Path(__file__).parent / "fixtures"
COLS = ["PassengerId", "Pclass", "Name", "Sex", "Ticket", "Fare", "Embarked", "Survived"]
NAN = np.nan


def frame(rows):
    return pd.DataFrame(rows, columns=COLS)


def family(n_women, pclass=3, boys=0, ticket="T1"):
    rows = [(i + 1, pclass, f"Fam, Mrs. W{i}", "female", ticket, 20.0, "S", NAN) for i in range(n_women)]
    rows += [(n_women + i + 1, pclass, f"Fam, Master. B{i}", "male", ticket, 20.0, "S", NAN) for i in range(boys)]
    return frame(rows)


def test_key_builders_differ_as_declared():
    pool = frame(
        [
            (1, 2, "Ford, Mr. A", "male", "111", 10.0, "S", NAN),
            (2, 2, "Ford, Mrs. B", "female", "222", 99.0, "S", NAN),  # surname + class + port only
            (3, 2, "Gray, Mrs. C", "female", "7", 10.0, "S", NAN),  # one-char ticket: empty stem
            (4, 2, "Gray, Miss. D", "female", "8", 10.0, "S", NAN),
            (5, 1, "Ford, Mr. E", "male", "333", 10.0, "S", NAN),  # other class
            (6, 2, "Ford, Mr. F", "male", "112", 10.0, "S", NAN),  # same stem and fare as row 1
        ]
    )
    a, b, c = (build_groups(pool, keys=k) for k in "abc")
    assert a[0] == a[5] and a[1] != a[0] and a[2] != a[3]
    assert b[0] == b[5] and b[1] != b[0] and b[2] == b[3]  # name-only key tolerates the empty stem
    assert c[0] == c[1] == c[5] and c[2] == c[3] and c[4] != c[0] and c[0] != c[2]
    assert build_groups(pool, nearby=True)[1] != build_groups(pool, nearby=True)[0]
    with pytest.raises(ValueError):
        build_groups(pool, keys="z")


def test_woman_rate_threshold():
    pool = family(5, pclass=2, boys=1)  # women 0-4, boy 5
    g = build_groups(pool)
    all_died = np.array([NAN, 0, 0, 0, 0, NAN])
    assert apply_rules(pool, g, all_died)[1][0] == DIE
    quarter = np.array([NAN, 0, 0, 0, 1, NAN])  # 1 of 4 mates lived: rate 0.25
    assert apply_rules(pool, g, quarter)[1][0] == DEFAULT
    pred, reason = apply_rules(pool, g, quarter, woman_rate=True)
    assert reason[0] == DIE and pred[0] == 0
    one = np.array([NAN, 0, NAN, NAN, NAN, NAN])  # a single known mate: only the all-died rule fires
    assert apply_rules(pool, g, one)[1][0] == DIE
    assert apply_rules(pool, g, one, woman_rate=True)[1][0] == DEFAULT
    third = np.array([NAN, 0, 0, 1, NAN, NAN])  # rate 1/3 > 0.25
    assert apply_rules(pool, g, third, woman_rate=True)[1][0] == DEFAULT


def test_boy_rate_threshold():
    pool = family(5, pclass=2, boys=1)
    g = build_groups(pool)
    three_of_four = np.array([1, 1, 1, 0, NAN, NAN])
    assert apply_rules(pool, g, three_of_four)[1][5] == DEFAULT
    pred, reason = apply_rules(pool, g, three_of_four, boy_rate=True)
    assert reason[5] == LIVE and pred[5] == 1
    half = np.array([1, 1, 0, 0, NAN, NAN])
    assert apply_rules(pool, g, half, boy_rate=True)[1][5] == DEFAULT
    one = np.array([1, NAN, NAN, NAN, NAN, NAN])
    assert apply_rules(pool, g, one)[1][5] == LIVE
    assert apply_rules(pool, g, one, boy_rate=True)[1][5] == DEFAULT


def test_class3_large_rule():
    big = family(5)
    g = build_groups(big)
    unknown = np.full(5, NAN)
    pred, reason = apply_rules(big, g, unknown, class3_large=True)
    assert set(reason) == {CLASS3} and not pred.any()
    assert apply_rules(big, g, unknown)[0].all()  # default: women live
    one_known = np.array([NAN, 1, NAN, NAN, NAN])  # one known mate: rule inactive for the others
    assert apply_rules(big, g, one_known, class3_large=True)[1][0] == DEFAULT
    small = family(4)
    assert set(apply_rules(small, build_groups(small), np.full(4, NAN), class3_large=True)[1]) == {DEFAULT}
    first = family(5, pclass=1)
    assert set(apply_rules(first, build_groups(first), np.full(5, NAN), class3_large=True)[1]) == {DEFAULT}


def seg_pool():
    return frame(
        [
            (1, 1, "Rich, Mr. A", "male", "A1", 50.0, "S", NAN),  # 1st-class man alone
            (2, 2, "Poor, Mr. B", "male", "B2", 10.0, "S", NAN),  # 2nd-class man
            (3, 3, "Kid, Master. C", "male", "C3", 8.0, "S", NAN),  # boy alone
            (4, 3, "Lone, Mrs. D", "female", "D4", 7.0, "S", NAN),  # 3rd-class woman alone
            (5, 2, "Mid, Mrs. E", "female", "E5", 12.0, "S", NAN),  # 2nd-class woman alone
            (6, 1, "Fam, Mr. F", "male", "F6", 60.0, "S", NAN),  # 1st-class man with a known mate
            (7, 1, "Fam, Mrs. G", "female", "F6", 60.0, "S", NAN),
        ]
    )


def test_segment_flips_restricted():
    pool = seg_pool()
    g = build_groups(pool)
    known = np.array([NAN] * 6 + [1.0])
    rule, reason = apply_rules(pool, g, known)
    out = segment_flips(pool, g, known, reason, rule, np.full(7, 0.9), 0.8, 0.2)
    assert list(out) == [1, 0, 1, 1, 1, 0, 1]  # only man A and boy C flip up; man F has a known mate
    out = segment_flips(pool, g, known, reason, rule, np.full(7, 0.1), 0.8, 0.2)
    assert list(out) == [0, 0, 0, 0, 1, 0, 1]  # only the lone 3rd-class woman flips down
    assert list(segment_flips(pool, g, known, reason, rule, np.full(7, NAN), 0.8, 0.2)) == list(rule)
    big = family(5)  # overridden rows are never touched
    gb = build_groups(big)
    r, rs = apply_rules(big, gb, np.full(5, NAN), class3_large=True)
    assert list(segment_flips(big, gb, np.full(5, NAN), rs, r, np.full(5, 0.1), 0.8, 0.2)) == list(r)


def test_own_label_never_affects_own_prediction_with_options():
    pool = pd.concat([seg_pool(), family(5), family(3, boys=2, ticket="T9")], ignore_index=True)
    g = build_groups(pool)
    rng = np.random.default_rng(1)
    prob = rng.random(len(pool))
    cfgs = [
        Config("a", woman_rate=True, boy_rate=True, class3_large=True, model="seg"),
        Config("c", class3_large=True, model="h3"),
    ]
    for cfg in cfgs:
        for _ in range(15):
            known = rng.integers(0, 2, len(pool)).astype(float)
            base, _ = predict_config(pool, g, known, cfg, prob)
            for i in range(len(pool)):
                flipped = known.copy()
                flipped[i] = 1 - flipped[i]
                assert predict_config(pool, g, flipped, cfg, prob)[0][i] == base[i]


def fixture_pool():
    train = pd.read_csv(FIXTURES / "train.csv")
    test = pd.read_csv(FIXTURES / "test.csv")
    return train, test, pool_passengers(train, test)


def test_grid_shapes_and_candidates():
    train, test, pool = fixture_pool()
    rng = np.random.default_rng(0)
    oof, tp = rng.random(len(train)), rng.random(len(test))
    cfgs = grid_configs(True)
    assert len(cfgs) <= 24 and len({c.name for c in cfgs}) == len(cfgs)
    assert len(grid_configs(False)) == 11
    ref = np.zeros(len(test), dtype=int)
    summary, preds = evaluate_grid(pool, len(train), oof, tp, ref, seeds=(42, 7), folds=2)
    assert len(summary) == len(cfgs) and set(preds) == set(summary["name"])
    assert list(summary["cv_mean"]) == sorted(summary["cv_mean"], reverse=True)
    assert {"name", "cv_mean", "cv_std", "cv_seed42", "n_diff_reference", "n_diff_wcg", "candidate"} <= set(summary.columns)
    assert all(len(p) == len(test) for p in preds.values())
    cand = summary[summary["candidate"] == "candidate"]
    assert cand["n_diff_reference"].between(2, 15).all()
    assert summary.loc[summary["n_seeds"] == 1, "name"].str.contains("h3|seg").all()
    no_ref, _ = evaluate_grid(pool, len(train), None, None, None, seeds=(42,), folds=2)
    assert len(no_ref) == 11 and (no_ref["candidate"] == "").all()


def test_run_grid_writes_validated_files(tmp_path):
    train, test, _ = fixture_pool()
    model = tmp_path / "m"
    model.mkdir()
    pd.DataFrame({"PassengerId": train["PassengerId"], "prob": 0.6}).to_csv(model / "oof.csv", index=False)
    pd.DataFrame({"PassengerId": test["PassengerId"], "prob": 0.6}).to_csv(model / "test_probs.csv", index=False)
    ref = tmp_path / "ref.csv"
    pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": 0}).to_csv(ref, index=False)
    summary = run_grid(FIXTURES, [model, model], ref, tmp_path / "sub", (42,), folds=2)
    refine = tmp_path / "sub" / "refine"
    assert {p.stem for p in refine.glob("*.csv")} == set(summary["name"]) | {"summary"}
    for name in summary["name"]:
        validate_submission(refine / f"{name}.csv", FIXTURES / "test.csv")
    assert len(pd.read_csv(refine / "summary.csv")) == len(summary)
