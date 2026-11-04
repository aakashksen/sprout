"""Run the T4.7 label-review stages in order. Each stage is one command:

  py tools/label_review.py regen-sample                  thumbs and table of the 50-photo field sample
  py tools/label_review.py sheets                        first-pass sheets, field_dev + field_test shuffled (r0001..)
  py tools/label_review.py train-sample [--n N]          first-pass sheets of a seeded field_train sample (s0001..)
  py tools/label_review.py collect --prefix r --reviews A.csv B.csv --out datasets/review/label_joined/first_r.csv
  py tools/label_review.py second-look --first datasets/review/label_joined/first_r.csv
  py tools/label_review.py collect --prefix c --reviews C.csv --out datasets/review/label_joined/second.csv
  py tools/label_review.py finish --first <joined first files> --second <joined second file> --version N
  py tools/label_review.py clean                         rewrite field_clean.csv from the exclusion log
  py tools/label_review.py e13                           score the cleaned field_dev rows (E13 records)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == "regen-sample":
        import pandas as pd

        from sprout.data.field import write_review

        write_review(pd.read_csv(ROOT / "datasets" / "manifests" / "field.csv"))
        return 0
    if cmd == "clean":
        from sprout.data.label_review import write_clean

        print(write_clean())
        return 0
    if cmd == "e13":
        from sprout.experiments.e13_clean_labels import main as e13

        return e13(rest)
    from sprout.data.label_review import main as review

    return review([cmd, *rest])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
