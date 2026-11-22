import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from voc.contracts import SCHEMAS, SanitizedInput, SourceKind
from voc.stats import (
    compute_stats,
    load_summary,
    save_summary,
    synthetic_stats,
    to_provenance,
    to_sanitized,
)


def chunk(ids, dates, flags):
    return pd.DataFrame({
        "ACCIDENT_NO": pd.array(ids, dtype="string"),
        "RCPT_YMD": pd.array(dates, dtype="string"),
        "is_air": pd.array(flags, dtype="boolean"),
    })


class StatsTest(unittest.TestCase):
    def test_months_denominators_and_duplicates(self):
        first, second = (
            chunk(["01", "02", "03"], ["20200201", "2020-01-31", "20200301"], [True, True, False]),
            chunk(["01", "04", None], ["20200101", "20200203", "20200230"], [True, None, True]),
        )
        a = compute_stats([first, second], "합성_A", "합성 A")
        b = compute_stats([second], "합성_B", "합성 B")
        self.assertEqual((a.total_rows, a.air_rows, a.undetermined_rows), (6, 4, 1))
        self.assertEqual((b.total_rows, b.air_rows, b.undetermined_rows), (3, 2, 1))
        self.assertEqual((a.period_start, a.period_end), ("2020-01-01", "2020-03-01"))
        self.assertEqual(a.monthly, (("2020-01", 2), ("2020-02", 1)))
        self.assertEqual((a.unique_air_ids, a.duplicate_air_rows), (2, 1))

    def test_summary_round_trip(self):
        results = synthetic_stats()
        with patch.object(Path, "write_text") as write:
            save_summary(results, "synthetic.json")
        with patch.object(Path, "read_text", return_value=write.call_args.args[0]):
            self.assertEqual(load_summary("synthetic.json"), results)

    def test_contract_conversions_and_synthetic_stats(self):
        results = synthetic_stats()
        self.assertEqual(len(results), 2)
        for stats in results:
            self.assertTrue(any("실제 수치가 아닙니다" in note for note in stats.notes))
            sanitized = to_sanitized(stats, SourceKind.SYNTHETIC)
            self.assertIsInstance(sanitized, SanitizedInput)
            self.assertEqual(sanitized.dataset_id, stats.dataset_id)
            fields = {key: value for key, value in sanitized.fields}
            self.assertEqual(fields.keys(), SCHEMAS["voc_aggregate_v1"].keys())
            assert stats.period_end is not None
            self.assertEqual(fields["기간"], stats.period_end[:7])
            self.assertEqual(fields["건수"], stats.air_rows)
            self.assertEqual(fields["비율"], stats.air_rows / stats.total_rows)
            provenance = to_provenance(stats)
            self.assertEqual(provenance.dataset, stats.dataset_id)
            self.assertEqual(provenance.period, f"{stats.period_start}~{stats.period_end}")
            self.assertIn("검증 대응표", provenance.filter_rule)
            self.assertEqual(provenance.denominator, stats.total_rows)
            self.assertEqual(provenance.dedup_basis, "ACCIDENT_NO")
            self.assertEqual(provenance.limitations, stats.notes)
        with self.assertRaises(ValueError):
            replace(results[0], dataset_id="invalid dataset")
        with self.assertRaises(ValueError):
            to_sanitized(replace(results[0], period_end=None), SourceKind.SYNTHETIC)
