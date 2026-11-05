import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TOOL = REPO / "tools" / "check_report_numbers.py"


def load():
    spec = importlib.util.spec_from_file_location("check_report_numbers", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


BULLETS = "## SPROUT - X\n\n1. Reached 0.771 macro-F1.\n\n## Evidence required\n\n| a | 0.999 |\n"


def make(tmp_path, report="Macro-F1 was 0.771.\n", bullets=BULLETS, claims=None, record=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "docs").mkdir(exist_ok=True)
    (tmp_path / "results" / "runs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "docs" / "REPORT.md").write_text(report, encoding="utf-8")
    (tmp_path / "docs" / "RESUME-TARGETS.md").write_text(bullets, encoding="utf-8")
    claims = [{"id": "c1"}] if claims is None else claims
    (tmp_path / "results" / "claims.json").write_text(json.dumps({"claims": claims}), encoding="utf-8")
    record = {"run_id": "E1-x", "n": 4106, "metrics": {"s": {"macro_f1": 0.7714, "err": 0.123}}, "config": {"lr": 0.0001}} \
        if record is None else record
    (tmp_path / "results" / "runs" / "E1-x.json").write_text(json.dumps(record), encoding="utf-8")
    return tmp_path


def problems(tmp_path, **kw):
    return load().check(make(tmp_path, **kw))


def test_clean_tree_passes(tmp_path):
    assert problems(tmp_path) == []


def test_invented_number_in_report_fails(tmp_path):
    out = problems(tmp_path, report="Macro-F1 was 0.812.\n")
    assert len(out) == 1 and "0.812" in out[0] and "REPORT.md:1" in out[0]


def test_invented_number_in_bullets_fails_but_other_sections_ignored(tmp_path):
    out = problems(tmp_path, bullets=BULLETS.replace("0.771", "0.5"))
    assert len(out) == 1 and "0.5" in out[0] and "RESUME-TARGETS" in out[0]


def test_rounding_matches_but_wrong_digit_does_not(tmp_path):
    assert problems(tmp_path, report="0.77 and 0.771\n") == []
    assert len(problems(tmp_path, report="0.772\n")) == 1


def test_percent_traces_to_fraction_in_record(tmp_path):
    assert problems(tmp_path, report="Error rate 12% and 77.1%.\n") == []
    assert len(problems(tmp_path, report="Error rate 13%.\n")) == 1


def test_percent_stored_as_percent_in_record(tmp_path):
    rec = {"run_id": "E1-x", "metrics": {"s": {"cut": 41.5}}}
    assert problems(tmp_path, report="Cut 41.5%.\n", bullets="## SPROUT - X\n", record=rec) == []


def test_sign_and_thousands_separator(tmp_path):
    assert problems(tmp_path, report="Change of -0.771 over 4,106 rows.\n") == []


def test_count_not_in_records_fails(tmp_path):
    assert len(problems(tmp_path, report="Used 4,107 rows.\n")) == 1


def test_config_values_count_as_record_values(tmp_path):
    assert problems(tmp_path, report="Learning rate 0.0001.\n") == []


def test_numbers_that_are_not_metrics_are_exempt(tmp_path):
    text = (
        "## 3.2 Data\n\n1. First\n- Task T8.2 and E12, G1, D5 (#90), DINOv2, ConvNeXt-T.\n"
        "Dated 2026-11-04, seed 42, top-3 and Recall@5, k=5, in `src/sprout/eval/x2.py` "
        "and docs/phase-4.md, see https://x.org/a/1234/5678. Python 3.12. Seven of 9 classes.\n"
    )
    assert problems(tmp_path, report=text) == []


def test_exemptions_do_not_cover_metric_shaped_numbers(tmp_path):
    for bad in ("`0.812`", "(0.812)", "T8.2: 0.812", "seed 42 scored 0.812", "# 0.812", "99%", "top-3 0.812", "1,234"):
        assert len(problems(tmp_path / bad[:3].encode().hex(), report=bad + "\n")) == 1, bad


def test_placeholder_fails_in_report_and_bullets(tmp_path):
    out = problems(tmp_path, report="Results: [placeholder]\n")
    assert any("placeholder" in p for p in out)
    out = problems(tmp_path / "b", bullets=BULLETS.replace("0.771", "[field_f1_after]"))
    assert any("[field_f1_after]" in p for p in out)


def test_markdown_link_is_not_a_placeholder(tmp_path):
    assert problems(tmp_path, report="See [the doc](docs/x.md) and [x].\n") == []


def test_empty_claims_fails(tmp_path):
    out = problems(tmp_path, claims=[])
    assert out == ["results/claims.json has no claims"]


def test_unreadable_claims_fails(tmp_path):
    root = make(tmp_path)
    (root / "results" / "claims.json").write_text("{nope", encoding="utf-8")
    assert any("claims.json" in p for p in load().check(root))


def test_claim_values_count_and_extra_claim_keys_are_tolerated(tmp_path):
    claims = [{"id": "c", "value": 0.4321, "provisional": True}]
    assert problems(tmp_path, report="Value 0.4321.\n", claims=claims) == []


def test_missing_report_fails(tmp_path):
    root = make(tmp_path)
    (root / "docs" / "REPORT.md").unlink()
    assert any("REPORT.md" in p for p in load().check(root))


def test_run_ids_and_hardware_model_names_are_not_numbers(tmp_path):
    ok = "See E17-20261105T024504962312Z and E14H-20261104T174311169630Z on a Ryzen AI 9 HX 370." + chr(10)
    assert problems(tmp_path, report=ok) == []
    assert len(problems(tmp_path, report="Run 20261105 and 370 ms." + chr(10))) == 2
