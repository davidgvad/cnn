from __future__ import annotations
import csv
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import export_paper_tables as analysis


class ResultsAnalysisTests(unittest.TestCase):
    def test_factorial_contrasts_of_known_polynomial(self):
        values = {name: 11 + 3*f + 5*b + 7*s - 4*f*b - 2*f*s - 6*b*s + 8*f*b*s
                  for name, (_, (f, b, s)) in analysis.CONFIGURATIONS.items()}
        effects = analysis.contrasts(values)
        self.assertEqual(effects["marginal_pp"], {"focal": 2., "batching": 2., "scaling": 5.})
        self.assertEqual(effects["pairwise_pp"], {"focal_batching": 0., "focal_scaling": 2., "batching_scaling": -2.})
        self.assertEqual(effects["three_way_pp"], 8.)

    def test_summary_needs_unique_cells_and_complete_seed_coverage(self):
        rows = []
        for architecture in analysis.scoring.ARCHITECTURES:
            for configuration in analysis.CONFIGURATIONS:
                rows.append({"architecture": architecture, "configuration": configuration,
                             "runs": 3, "seeds": "0,1,2",
                             **{metric + "_mean": .5 for metric in analysis.scoring.METRICS},
                             **{metric + "_std": .1 for metric in analysis.scoring.METRICS}})
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "summary.csv"
            analysis.write_csv(path, rows)
            summary = analysis.read_summary(path, (0, 1, 2))
            self.assertEqual(len(summary), 32)
            rows[-1] = rows[0].copy()
            analysis.write_csv(path, rows)
            with self.assertRaisesRegex(ValueError, "four backbones and eight configurations"):
                analysis.read_summary(path, (0, 1, 2))
            rows[-1] = {**rows[-1], "architecture": "mlp", "configuration": "full_retuned", "seeds": "0,0,2"}
            analysis.write_csv(path, rows)
            with self.assertRaisesRegex(ValueError, "Incomplete seed summary"):
                analysis.read_summary(path, (0, 1, 2))

    def test_export_preserves_unrounded_csv_and_formats_latex(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            analysis.export_table(directory, "table_example", "tab:example",
                                  [{"Configuration": "A_B", "Range": 10.445070089008656}])
            with (directory / "table_example.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(float(row["Range"]), 10.445070089008656)
            tex = (directory / "table_example.tex").read_text()
            self.assertIn("10.45", tex)
            self.assertIn(r"A\_B", tex)
            self.assertIn(r"\label{tab:example}", tex)


if __name__ == "__main__":
    unittest.main()
