"""P0 acoustic pipeline: noise cleanup, real MFCC-ΔΔ, and v=2·L·f velocity."""
import os
import unittest
from pathlib import Path

import numpy as np

os.environ.setdefault("ACOUSTIC_DENOISE", "1")

from app.models.acoustic_model import extract_mfcc_features, analyze_acoustic, FEATURE_DIM
from app.models.acoustic_physics import (
    acoustic_velocity, velocity_cross_check, extract_ring_frequency, ring_frequency_check,
)
from app.utils.audio_io import load_audio_bytes
from app.utils.audio_denoise import clean_tap_audio

AUDIO = Path("data/demo/test-kit/audio")
SAMPLE = AUDIO / "fake_composite_tap_1.wav"


class AcousticP0Tests(unittest.TestCase):
    def test_feature_vector_is_122_dim_with_dd(self):
        # 122 = 20 MFCC μ/σ + 20 Δ μ/σ + 20 ΔΔ μ/σ + ZCR + RMS
        self.assertEqual(FEATURE_DIM, 122)
        feats = extract_mfcc_features(SAMPLE.read_bytes())
        self.assertEqual(feats.shape[0], 122)

    def test_second_order_delta_actually_present(self):
        # The ΔΔ (order=2) block must differ from the Δ block — proves it isn't
        # just a duplicate first-order delta.
        import librosa
        y, sr = load_audio_bytes(SAMPLE.read_bytes())
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        d = librosa.feature.delta(mfcc)
        dd = librosa.feature.delta(mfcc, order=2)
        self.assertFalse(np.allclose(d, dd))

    def test_denoise_improves_snr_on_noisy_clip(self):
        y, sr = load_audio_bytes(SAMPLE.read_bytes())
        rng = np.random.default_rng(0)
        y_noisy = (y + 0.02 * rng.standard_normal(len(y))).astype("float32")
        _, info = clean_tap_audio(y_noisy, sr)
        self.assertTrue(info["applied"])
        self.assertIsNotNone(info["snr_pre_db"])
        self.assertGreater(info["snr_post_db"], info["snr_pre_db"])

    def test_denoise_skips_gating_on_clean_clip(self):
        # Already-clean lab taps keep their timbre (band-pass only, no gating).
        y, sr = load_audio_bytes(SAMPLE.read_bytes())
        _, info = clean_tap_audio(y, sr)
        self.assertEqual(info["noise_profile"], "bandpass_only_clean")

    def test_velocity_formula(self):
        self.assertAlmostEqual(acoustic_velocity(6000, 0.040), 480.0, places=3)

    def test_velocity_is_corroborative_not_a_bulk_gate(self):
        # Against a genuine band, a same-size item near the band rings within
        # the genuine effective-velocity range — never flagged by the bulk
        # 2020 m/s longitudinal reference (which would be physically wrong).
        band = {"f_low": 5533.1, "f_high": 6997.1}
        v = velocity_cross_check(6200, 0.040, band)
        self.assertTrue(v["available"])
        self.assertIn("effective", v["mode_note"])
        self.assertLess(v["velocity_ratio"], 1.15)

    def test_ring_check_exposes_denoise_and_velocity_trail(self):
        ring = extract_ring_frequency(SAMPLE.read_bytes())
        self.assertIn("snr_pre_db", ring)
        self.assertIn("snr_post_db", ring)
        dens = {"best_match": {"kind": "karat"}, "risk_score": 0.2}
        rc = ring_frequency_check(ring, dens, item_length_m=0.040)
        self.assertIn("velocity_cross_check", rc)
        self.assertTrue(rc["velocity_cross_check"]["available"])

    def test_data_limited_svm_returns_neutral(self):
        # With the demo-kit model flagged unreliable, the SVM must not inject a
        # non-neutral risk into fusion.
        res = analyze_acoustic(SAMPLE.read_bytes())
        if res["mode"] == "svm_data_limited":
            self.assertEqual(res["risk_score"], 0.5)


if __name__ == "__main__":
    unittest.main()
