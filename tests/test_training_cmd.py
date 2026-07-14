"""Kommandoaufbau des Trainings-Managers.

Fixiert insbesondere den Batch-Bugfix: word-(1,2) in der Analyzer-Auswahl muss
--word-ngram-max 2 an den Subprozess uebergeben (vorher wurde im Batch immer 1
erzwungen, d. h. Wort-Bigramme wurden nie trainiert).
"""
import pytest

from pdl_lt_sg_predict.core import training as tr


@pytest.fixture
def manager(monkeypatch, tmp_path):
    mgr = tr.TrainingManager()
    csv = tmp_path / "daten.csv"
    csv.write_text("lemma,bedeutung,sachgruppe\nHaus,Gebaeude,1\n", encoding="utf-8")
    mgr.csv_path = csv
    monkeypatch.setattr(tr, "ENABLE_TRAINING", True)
    captured: dict = {}

    def fake_launch(cmd, progress_file, *, mode, estimated_fit_sec=1.0, total=1):
        captured["cmd"] = cmd
        captured["total"] = total

    monkeypatch.setattr(mgr, "_launch", fake_launch)
    mgr.captured = captured
    return mgr


def _flag_value(cmd: list[str], flag: str) -> str:
    return cmd[cmd.index(flag) + 1]


def test_batch_word_bigrams_reach_subprocess(manager):
    result = manager.start_batch({
        "batch_model_types": ["svm"],
        "batch_use_stopwords": [False],
        "batch_min_lengths": [1],
        "batch_analyzers": ["char_wb", "word-(1,1)", "word-(1,2)"],
    })
    cmd = manager.captured["cmd"]
    assert _flag_value(cmd, "--word-ngram-max") == "2"
    # word-(1,1) und word-(1,2) verschmelzen zu EINER word-Konfiguration.
    i = cmd.index("--analyzer")
    assert cmd[i + 1:i + 3] == ["char_wb", "word"]
    assert result["total"] == 2
    assert manager.captured["total"] == 2


def test_batch_char_wb_only_keeps_ngram_1(manager):
    manager.start_batch({
        "batch_model_types": ["svm"],
        "batch_use_stopwords": [False],
        "batch_min_lengths": [1],
        "batch_analyzers": ["char_wb"],
    })
    assert _flag_value(manager.captured["cmd"], "--word-ngram-max") == "1"


def test_single_char_wb_forces_ngram_1(manager):
    manager.start_single({
        "model": "svm", "analyzer_mode": "char_wb", "word_ngram_max": 2,
    })
    cmd = manager.captured["cmd"]
    assert _flag_value(cmd, "--word-ngram-max") == "1"
    assert _flag_value(cmd, "--analyzer") == "char_wb"


def test_single_word_passes_ngram_through(manager):
    manager.start_single({
        "model": "svm", "analyzer_mode": "word", "word_ngram_max": 2,
    })
    cmd = manager.captured["cmd"]
    assert _flag_value(cmd, "--word-ngram-max") == "2"
    assert _flag_value(cmd, "--analyzer") == "word"
