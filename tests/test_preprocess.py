"""Unit tests for image preprocessing utilities."""
from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

try:  # Streamlit is optional for running the unit tests locally
    import app  # type: ignore[import]
    APP_IMPORT_ERROR: str | None = None
except Exception as exc:  # pragma: no cover - handled by skipping tests
    app = None  # type: ignore[assignment]
    APP_IMPORT_ERROR = str(exc)


class PreprocessImageTests(unittest.TestCase):
    """Validate that preprocessing prepares images for the CNN."""

    @unittest.skipIf(app is None, f"app module unavailable: {APP_IMPORT_ERROR}")
    def test_output_shape_and_dtype(self) -> None:
        image = Image.new("RGB", (120, 160), color="white")

        processed = app.preprocess_image(image)

        self.assertEqual(processed.shape, (1, 48, 48, 1))
        self.assertEqual(processed.dtype, np.float32)

    @unittest.skipIf(app is None, f"app module unavailable: {APP_IMPORT_ERROR}")
    def test_values_are_normalised_between_zero_and_one(self) -> None:
        image = Image.new("RGB", (48, 48), color="white")

        processed = app.preprocess_image(image)

        self.assertTrue(np.all(processed >= 0.0))
        self.assertTrue(np.all(processed <= 1.0))
        self.assertAlmostEqual(float(processed.mean()), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
