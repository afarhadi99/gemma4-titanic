import os

import pytest

from titanic.env import load_env, missing


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in ("T_A", "T_B", "T_C", "T_D", "T_E", "T_EMPTY"):
        monkeypatch.delenv(key, raising=False)


def write(tmp_path, text):
    path = tmp_path / ".env"
    path.write_text(text, encoding="utf-8")
    return path


def test_parses_comments_blanks_quotes_and_export(tmp_path):
    path = write(
        tmp_path,
        "# comment\n\nT_A=1\nT_B=\"two words\"\nexport T_C='3'\n  T_D = spaced \nbad line\nT_E=a=b\n",
    )
    loaded = load_env(path)
    assert loaded == {"T_A": "1", "T_B": "two words", "T_C": "3", "T_D": "spaced", "T_E": "a=b"}
    assert os.environ["T_B"] == "two words"


def test_does_not_override_existing_by_default(tmp_path, monkeypatch):
    monkeypatch.setenv("T_A", "keep")
    path = write(tmp_path, "T_A=new\nT_B=2\n")
    assert load_env(path) == {"T_B": "2"}
    assert os.environ["T_A"] == "keep"


def test_override(tmp_path, monkeypatch):
    monkeypatch.setenv("T_A", "old")
    assert load_env(write(tmp_path, "T_A=new\n"), override=True) == {"T_A": "new"}
    assert os.environ["T_A"] == "new"


def test_missing_file_returns_empty(tmp_path):
    assert load_env(tmp_path / "nope.env") == {}


def test_mismatched_quotes_kept(tmp_path):
    assert load_env(write(tmp_path, "T_A=\"abc'\n")) == {"T_A": "\"abc'"}


def test_missing_reports_unset_and_empty(monkeypatch):
    monkeypatch.setenv("T_A", "x")
    monkeypatch.setenv("T_EMPTY", "")
    assert missing(["T_A", "T_B", "T_EMPTY"]) == ["T_B", "T_EMPTY"]
