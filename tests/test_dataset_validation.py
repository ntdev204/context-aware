from __future__ import annotations

import json
import tempfile
from pathlib import Path

from scripts.data.validate_dataset import validate


class TestDatasetValidation:
    def test_warns_when_report_predates_review_gate_fields(self, capsys):
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "report.json"
            report = {
                "total_images": 1000,
                "classes": {
                    "STATIONARY": 300,
                    "APPROACHING": 200,
                    "DEPARTING": 200,
                    "CROSSING": 150,
                    "ERRATIC": 150,
                },
                "duplicates": [],
                "corrupt_images": [],
            }
            report_path.write_text(json.dumps(report), encoding="utf-8")

            status = validate(report_path)
            captured = capsys.readouterr()

            assert status == 0
            assert "predates review-gate fields" in captured.out
