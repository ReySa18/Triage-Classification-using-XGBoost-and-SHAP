import sys
import unittest
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
APP_DIR = ROOT_DIR / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from backend.feature_engineering import (  # noqa: E402
    FEATURE_COLS,
    REQUIRED_INPUT_FIELDS,
    IncompleteInputError,
    build_feature_vector,
    get_missing_required_fields,
    validate_required_inputs,
)
from backend.predictor import predict_triage_v3  # noqa: E402


def complete_input() -> dict:
    return {
        "usia_tahun": 35,
        "jenis_kelamin": "Laki-laki",
        "cara_datang": "Sendiri",
        "GCS_E": 4,
        "GCS_M": 6,
        "GCS_V": 5,
        "sistole": 120,
        "diastole": 80,
        "denyut_jantung": 80,
        "laju_pernafasan": 16,
        "suhu_tubuh": 36.5,
        "SpO2": 98,
    }


class RequiredInputValidationTests(unittest.TestCase):
    def test_complete_input_has_no_missing_fields(self):
        input_data = complete_input()

        self.assertEqual(get_missing_required_fields(input_data), [])
        self.assertIsNone(validate_required_inputs(input_data))

    def test_missing_keys_and_empty_values_are_reported_in_field_order(self):
        input_data = complete_input()
        input_data["jenis_kelamin"] = "  "
        input_data["GCS_E"] = None
        input_data["suhu_tubuh"] = np.nan
        del input_data["SpO2"]

        self.assertEqual(
            get_missing_required_fields(input_data),
            ["jenis_kelamin", "GCS_E", "suhu_tubuh", "SpO2"],
        )

    def test_validation_error_exposes_keys_and_user_labels(self):
        input_data = complete_input()
        input_data["usia_tahun"] = None
        input_data["SpO2"] = np.nan

        with self.assertRaises(IncompleteInputError) as context:
            validate_required_inputs(input_data)

        self.assertEqual(context.exception.missing_fields, ("usia_tahun", "SpO2"))
        self.assertEqual(context.exception.missing_labels, ("Usia (tahun)", "SpO₂"))
        self.assertIn("Usia (tahun)", str(context.exception))
        self.assertIn("SpO₂", str(context.exception))

    def test_empty_dictionary_is_rejected_instead_of_receiving_defaults(self):
        with self.assertRaises(IncompleteInputError) as context:
            build_feature_vector({})

        expected_fields = tuple(field for field, _label in REQUIRED_INPUT_FIELDS)
        self.assertEqual(context.exception.missing_fields, expected_fields)

    def test_complete_input_builds_ordered_31_feature_vector(self):
        feature_vector = build_feature_vector(complete_input())

        self.assertEqual(feature_vector.shape, (1, 31))
        self.assertEqual(feature_vector.columns.tolist(), FEATURE_COLS)
        self.assertFalse(feature_vector.isna().any().any())

    def test_zero_is_treated_as_provided_input(self):
        input_data = complete_input()
        input_data["laju_pernafasan"] = 0
        input_data["SpO2"] = 0

        self.assertEqual(get_missing_required_fields(input_data), [])
        feature_vector = build_feature_vector(input_data)
        self.assertEqual(feature_vector.at[0, "laju_pernafasan"], 0)
        self.assertEqual(feature_vector.at[0, "SpO2"], 0)

    def test_prediction_rejects_incomplete_input_before_artifact_access(self):
        with self.assertRaises(IncompleteInputError):
            predict_triage_v3({}, artifacts={})


if __name__ == "__main__":
    unittest.main()
