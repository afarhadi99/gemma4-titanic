import json
from pathlib import Path

import numpy as np
import pandas as pd

from titanic.submission import validate_submission
from titanic.wcg import DEFAULT, DIE, LIVE, UnionFind, apply_rules, build_groups, combine, link_keys
from wcg import run

FIXTURES = Path(__file__).parent / "fixtures"


def frame(rows):
    cols = ["PassengerId", "Pclass", "Name", "Sex", "Ticket", "Fare", "Embarked", "Survived"]
    return pd.DataFrame(rows, columns=cols)


def tiny():
    return frame(
        [
            (1, 3, "Smith, Master. Tom", "male", "A1", 10.0, "S", np.nan),
            (2, 3, "Smith, Mrs. Ann", "female", "A1", 10.0, "S", np.nan),
            (3, 3, "Jones, Mrs. Bea", "female", "B2", 8.0, "S", np.nan),
            (4, 3, "Jones, Miss. Cy", "female", "B2", 8.0, "S", np.nan),
            (5, 3, "Lone, Mr. Dan", "male", "C3", 7.0, "S", np.nan),
            (6, 3, "Free, Mrs. Eve", "female", "D4", 9.0, "S", np.nan),
        ]
    )


def test_union_find_links_transitively():
    uf = UnionFind(5)
    link_keys(["a", "b", "a", None, "b"], uf)
    assert uf.find(0) == uf.find(2) and uf.find(1) == uf.find(4)
    assert uf.find(0) != uf.find(1) and uf.find(3) == 3
    link_keys(["x", "x", "y", "y", None], uf)
    assert uf.find(0) == uf.find(1)  # joined via the second key


def test_build_groups_ticket_family_and_nearby():
    pool = frame(
        [
            (1, 2, "Ford, Mr. A", "male", "W./C. 6608", 34.4, "S", np.nan),
            (2, 2, "Ford, Mrs. B", "female", "W./C. 6608", 34.4, "S", np.nan),
            (3, 2, "Cole, Mr. C", "male", "12345", 20.0, "S", np.nan),
            (4, 2, "Cole, Mrs. D", "female", "12346", 20.0, "S", np.nan),  # same stem + fare
            (5, 2, "Cole, Miss. E", "female", "12347", 21.0, "S", np.nan),  # stem only
        ]
    )
    g = build_groups(pool)
    assert g[0] == g[1] and g[2] == g[3] and g[0] != g[2] and g[4] != g[2]
    g2 = build_groups(pool, nearby=True)
    assert g2[2] == g2[3] == g2[4]


def test_rules_on_hand_made_frame():
    pool = tiny()
    groups = build_groups(pool)
    known = np.array([np.nan, 1, 0, 0, np.nan, np.nan])
    pred, reason = apply_rules(pool, groups, known)
    assert pred[0] == 1 and reason[0] == LIVE  # boy with surviving mother
    assert pred[2] == 0 and reason[2] == DIE  # sister died, nobody else known
    assert pred[3] == 0 and reason[3] == DIE
    assert pred[4] == 0 and reason[4] == DEFAULT  # man dies
    assert pred[5] == 1 and reason[5] == DEFAULT  # woman alone: gender default
    assert pred[1] == 1 and reason[1] == DEFAULT  # mother: boy's label unknown
    known[0] = 0  # boy died -> mother's rule now fires
    pred, reason = apply_rules(pool, groups, known)
    assert pred[1] == 0 and reason[1] == DIE and pred[0] == 1 and reason[0] == LIVE


def test_boy_dies_unless_all_mates_survived():
    pool = tiny()
    groups = build_groups(pool)
    pred, reason = apply_rules(pool, groups, np.array([np.nan, 0, np.nan, np.nan, np.nan, np.nan]))
    assert pred[0] == 0 and reason[0] == DEFAULT


def test_own_label_never_affects_own_prediction():
    pool = tiny()
    groups = build_groups(pool)
    rng = np.random.default_rng(0)
    for _ in range(20):
        known = rng.integers(0, 2, len(pool)).astype(float)
        base, base_reason = apply_rules(pool, groups, known)
        for i in range(len(pool)):
            flipped = known.copy()
            flipped[i] = 1 - flipped[i]
            pred, reason = apply_rules(pool, groups, flipped)
            assert pred[i] == base[i] and reason[i] == base_reason[i]


def test_combine_variants():
    pool = tiny()
    reason = np.array([LIVE, DEFAULT, DIE, DIE, DEFAULT, DEFAULT], dtype=object)
    rule = np.array([1, 1, 0, 0, 0, 1])
    prob = np.array([0.1, 0.1, 0.9, 0.9, 0.9, 0.1])
    assert list(combine("h1", pool, rule, reason)) == [1, 1, 0, 0, 0, 1]
    assert list(combine("h2", pool, rule, reason, prob)) == [1, 0, 0, 0, 1, 0]
    # h3: the confident man lives, the confident-die woman dies; overridden rows untouched
    assert list(combine("h3", pool, rule, reason, prob)) == [1, 0, 0, 0, 1, 0]
    assert list(combine("gender", pool, rule, reason)) == [0, 1, 1, 1, 0, 1]


def test_run_writes_valid_submissions(tmp_path):
    train = pd.read_csv(FIXTURES / "train.csv")
    test = pd.read_csv(FIXTURES / "test.csv")
    model = tmp_path / "model"
    model.mkdir()
    pd.DataFrame({"PassengerId": train["PassengerId"], "prob": 0.5}).to_csv(model / "oof.csv", index=False)
    pd.DataFrame({"PassengerId": test["PassengerId"], "prob": 0.5}).to_csv(model / "test_probs.csv", index=False)
    out = tmp_path / "sub"
    result = run(FIXTURES, model, out, nearby=True, seeds=(42,), folds=2)
    for name in ("wcg", "wcg_h1", "wcg_h2", "wcg_h3"):
        validate_submission(out / f"{name}.csv", FIXTURES / "test.csv")
    assert set(result["files"]) == {"wcg", "h1", "h2", "h3"}
    assert json.loads((out / "wcg.json").read_text())["test_counts"]["wcg_overrides"] >= 0
