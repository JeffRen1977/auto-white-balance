"""
Pytest tests for the Auto White Balance system.

Tests cover the core AWB system: initialization, analyzers, fusion,
temporal filtering, and gain computation. Uses synthetic data so no
external files are needed.
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.awb_system import (
    AutoWhiteBalanceSystem,
    FrameStatsExtractor,
    HeuristicAnalyzer,
    FaceAwareAnalyzer,
    SegmentationAnalyzer,
    MLAnalyzer,
    FrameSummary,
    safe_divide,
)


def make_frame_summary(r_mean=500.0, g_mean=500.0, b_mean=500.0, grid_size=(8, 8)):
    """Create a synthetic FrameSummary for testing."""
    r_grid = np.full(grid_size, r_mean, dtype=np.float32)
    g_grid = np.full(grid_size, g_mean, dtype=np.float32)
    b_grid = np.full(grid_size, b_mean, dtype=np.float32)
    return FrameSummary(
        r_grid=r_grid,
        g_grid=g_grid,
        b_grid=b_grid,
        bv=5.0,
        scene="auto",
        face_rois=None,
        segmentation_map=None,
        spectral_data=None,
        depth_info=None,
    )


class TestSafeDivide:
    def test_basic_division(self):
        a = np.array([10.0, 20.0])
        b = np.array([2.0, 5.0])
        result = safe_divide(a, b)
        np.testing.assert_allclose(result, [5.0, 4.0], atol=1e-5)

    def test_zero_denominator(self):
        a = np.array([10.0])
        b = np.array([0.0])
        result = safe_divide(a, b)
        assert np.isfinite(result[0])


class TestHeuristicAnalyzer:
    def test_init(self):
        analyzer = HeuristicAnalyzer(name="test_heuristic")
        assert analyzer.name == "test_heuristic"

    def test_neutral_scene(self):
        """A neutral scene (equal R,G,B) should produce gains close to 1.0."""
        analyzer = HeuristicAnalyzer(name="gray_world")
        frame = make_frame_summary(r_mean=500.0, g_mean=500.0, b_mean=500.0)
        result = analyzer.analyze(frame)
        assert result is not None
        assert len(result.gains) == 3
        np.testing.assert_allclose(result.gains, [1.0, 1.0, 1.0], atol=0.15)

    def test_warm_scene(self):
        """A warm scene (high red, low blue) should boost blue gain."""
        analyzer = HeuristicAnalyzer(name="gray_world")
        frame = make_frame_summary(r_mean=700.0, g_mean=500.0, b_mean=300.0)
        result = analyzer.analyze(frame)
        assert result is not None
        assert result.gains[2] > result.gains[0]

    def test_trust_in_range(self):
        analyzer = HeuristicAnalyzer(name="gray_world")
        frame = make_frame_summary()
        result = analyzer.analyze(frame)
        assert 0.0 <= result.trust <= 1.0


class TestAutoWhiteBalanceSystem:
    def test_init_default(self):
        awb = AutoWhiteBalanceSystem()
        assert awb is not None

    def test_process_frame(self):
        awb = AutoWhiteBalanceSystem()
        frame = make_frame_summary()
        result = awb.process_frame(frame)
        assert result is not None
        assert len(result.gains) == 3
        assert result.cct > 0
        assert 0.0 <= result.trust <= 1.0

    def test_gains_are_positive(self):
        awb = AutoWhiteBalanceSystem()
        frame = make_frame_summary(r_mean=300.0, g_mean=500.0, b_mean=700.0)
        result = awb.process_frame(frame)
        assert all(g > 0 for g in result.gains)

    def test_processing_time_recorded(self):
        awb = AutoWhiteBalanceSystem()
        frame = make_frame_summary()
        result = awb.process_frame(frame)
        assert result.processing_time >= 0

    def test_ccm_shape(self):
        awb = AutoWhiteBalanceSystem()
        frame = make_frame_summary()
        result = awb.process_frame(frame)
        assert result.ccm.shape == (3, 3)

    def test_temporal_consistency(self):
        """Processing the same frame twice should give similar results."""
        awb = AutoWhiteBalanceSystem()
        frame = make_frame_summary()
        result1 = awb.process_frame(frame)
        result2 = awb.process_frame(frame)
        np.testing.assert_allclose(result1.gains, result2.gains, atol=0.3)

    def test_analyzer_results_present(self):
        awb = AutoWhiteBalanceSystem()
        frame = make_frame_summary()
        result = awb.process_frame(frame)
        assert len(result.analyzer_results) > 0


class TestFrameStatsExtractor:
    def test_extract_from_image(self):
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        extractor = FrameStatsExtractor()
        frame = extractor.extract_from_image(img, bv=5.0, scene="auto")
        assert frame is not None
        assert frame.r_grid is not None
        assert frame.g_grid is not None
        assert frame.b_grid is not None
