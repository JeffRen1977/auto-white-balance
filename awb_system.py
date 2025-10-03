"""
Advanced Auto White Balance (AWB) System

A comprehensive AWB implementation inspired by hybrid approaches, featuring:
- Multi-analyzer framework (heuristic, ML, face-aware, segmentation)
- Trust-based fusion with temporal smoothing
- CCM selection and spatial color correction
- Real-time processing optimization

Author: Auto White Balance System
Date: 2024
"""

import numpy as np
import cv2
from typing import Dict, Tuple, List, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import json
import time

# =============================================================================
# Utility Functions
# =============================================================================

def safe_divide(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Safe division with epsilon to avoid division by zero."""
    return a / (b + eps)

def rgb_to_chromaticity(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert RGB to chromaticity coordinates."""
    s = r + g + b + 1e-12
    return r / s, g / s, b / s

def estimate_cct_from_chromaticity(r: float, g: float, b: float) -> float:
    """
    Estimate Correlated Color Temperature (CCT) from chromaticity.
    
    This is a simplified estimation. For production use, implement
    proper Planckian locus mapping or use trained ML models.
    """
    x, y, _ = rgb_to_chromaticity(r, g, b)
    # Simple mapping: blue ratio -> CCT
    blue_ratio = b / (r + g + b + 1e-8)
    return 2000 + 8000 * np.clip(blue_ratio, 0, 1)

def compute_awb_gains(r_mean: float, g_mean: float, b_mean: float, 
                     target_g: float = 0.5) -> np.ndarray:
    """
    Compute AWB gains to normalize color channels.
    
    Args:
        r_mean, g_mean, b_mean: Mean values for each channel
        target_g: Target green ratio (default 0.5 for gray world)
    
    Returns:
        Array of [R_gain, G_gain, B_gain]
    """
    total = r_mean + g_mean + b_mean + 1e-12
    desired_g = total * target_g
    
    g_gain = safe_divide(desired_g, g_mean)
    r_gain = safe_divide(desired_g, r_mean)
    b_gain = safe_divide(desired_g, b_mean)
    
    # Normalize so G gain = 1.0
    r_gain /= g_gain
    b_gain /= g_gain
    g_gain = 1.0
    
    return np.array([r_gain, g_gain, b_gain], dtype=np.float32)

# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class FrameSummary:
    """Container for frame statistics and metadata."""
    r_grid: np.ndarray          # Red channel statistics (subsampled grid)
    g_grid: np.ndarray          # Green channel statistics
    b_grid: np.ndarray          # Blue channel statistics
    bv: float = 0.0            # Brightness Value from Auto Exposure
    scene: Optional[str] = None # Scene type from scene detection
    face_rois: Optional[List[Tuple[int, int, int, int]]] = None  # Face regions (x,y,w,h)
    segmentation_map: Optional[np.ndarray] = None  # Segmentation map
    spectral_data: Optional[Dict] = None  # Spectral sensor data
    depth_info: Optional[Dict] = None     # Depth information
    timestamp: float = 0.0      # Frame timestamp
    
    def global_means(self) -> Tuple[float, float, float]:
        """Compute global mean values for each channel."""
        return (float(np.mean(self.r_grid)), 
                float(np.mean(self.g_grid)), 
                float(np.mean(self.b_grid)))
    
    def region_means(self, mask: np.ndarray) -> Tuple[float, float, float]:
        """Compute mean values for a specific region."""
        if mask.shape != self.r_grid.shape:
            raise ValueError("Mask shape must match grid shape")
        return (float(np.mean(self.r_grid[mask])),
                float(np.mean(self.g_grid[mask])),
                float(np.mean(self.b_grid[mask])))

@dataclass
class AnalyzerResult:
    """Result from an AWB analyzer."""
    gains: np.ndarray    # AWB gains [R, G, B]
    cct: float          # Estimated CCT
    trust: float        # Confidence score [0, 1]
    tag: str = ""       # Analyzer identifier
    metadata: Dict = None  # Additional metadata
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class AWBSystemOutput:
    """Final AWB system output."""
    gains: np.ndarray           # Final AWB gains
    cct: float                 # Estimated CCT
    ccm: np.ndarray           # Color Correction Matrix
    trust: float              # Overall confidence
    analyzer_results: List[AnalyzerResult]  # Individual analyzer results
    processing_time: float    # Processing time in ms
    metadata: Dict           # Additional metadata
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'gains': self.gains.tolist(),
            'cct': self.cct,
            'ccm': self.ccm.tolist(),
            'trust': self.trust,
            'analyzer_results': [
                {
                    'gains': r.gains.tolist(),
                    'cct': r.cct,
                    'trust': r.trust,
                    'tag': r.tag,
                    'metadata': r.metadata
                }
                for r in self.analyzer_results
            ],
            'processing_time': self.processing_time,
            'metadata': self.metadata
        }

# =============================================================================
# Frame Statistics Extractor
# =============================================================================

class FrameStatsExtractor:
    """Extract frame statistics from images or sensor data."""
    
    def __init__(self, grid_width: int = 64, grid_height: int = 48):
        self.grid_width = grid_width
        self.grid_height = grid_height
    
    def extract_from_image(self, image: np.ndarray, **kwargs) -> FrameSummary:
        """
        Extract statistics from a full-resolution image.
        
        Args:
            image: Input image (BGR format)
            **kwargs: Additional metadata (bv, scene, face_rois, etc.)
        """
        # Resize to grid size for efficient processing
        small_image = cv2.resize(
            image, (self.grid_width, self.grid_height), 
            interpolation=cv2.INTER_AREA
        ).astype(np.float32)
        
        # Split channels (BGR -> RGB)
        b_channel = small_image[..., 0]
        g_channel = small_image[..., 1] 
        r_channel = small_image[..., 2]
        
        return FrameSummary(
            r_grid=r_channel,
            g_grid=g_channel,
            b_grid=b_channel,
            bv=kwargs.get('bv', 0.0),
            scene=kwargs.get('scene'),
            face_rois=kwargs.get('face_rois'),
            segmentation_map=kwargs.get('segmentation_map'),
            spectral_data=kwargs.get('spectral_data'),
            depth_info=kwargs.get('depth_info'),
            timestamp=kwargs.get('timestamp', time.time())
        )
    
    def extract_from_bayer_stats(self, r_stats: np.ndarray, g_stats: np.ndarray, 
                                b_stats: np.ndarray, **kwargs) -> FrameSummary:
        """
        Extract statistics from precomputed Bayer statistics.
        
        Args:
            r_stats, g_stats, b_stats: Precomputed channel statistics
            **kwargs: Additional metadata
        """
        return FrameSummary(
            r_grid=r_stats.astype(np.float32),
            g_grid=g_stats.astype(np.float32),
            b_grid=b_stats.astype(np.float32),
            bv=kwargs.get('bv', 0.0),
            scene=kwargs.get('scene'),
            face_rois=kwargs.get('face_rois'),
            segmentation_map=kwargs.get('segmentation_map'),
            spectral_data=kwargs.get('spectral_data'),
            depth_info=kwargs.get('depth_info'),
            timestamp=kwargs.get('timestamp', time.time())
        )

# =============================================================================
# Analyzer Base Class
# =============================================================================

class AWBAnalyzer(ABC):
    """Base class for AWB analyzers."""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def analyze(self, frame: FrameSummary) -> AnalyzerResult:
        """Analyze frame and return AWB decision."""
        pass
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"

# =============================================================================
# Analyzer Implementations
# =============================================================================

class HeuristicAnalyzer(AWBAnalyzer):
    """
    Heuristic-based AWB analyzer using gray world assumption
    and daylight locus principles.
    """
    
    def __init__(self, name: str = "heuristic"):
        super().__init__(name)
        self.gray_world_target = 0.5
    
    def analyze(self, frame: FrameSummary) -> AnalyzerResult:
        """Analyze using gray world and daylight locus heuristics."""
        r_mean, g_mean, b_mean = frame.global_means()
        
        # Gray world assumption
        gains = compute_awb_gains(r_mean, g_mean, b_mean, self.gray_world_target)
        
        # Estimate CCT
        cct = estimate_cct_from_chromaticity(r_mean, g_mean, b_mean)
        
        # Trust based on brightness and scene stability
        trust = self._compute_trust(frame, gains)
        
        return AnalyzerResult(
            gains=gains,
            cct=cct,
            trust=trust,
            tag=self.name,
            metadata={'method': 'gray_world', 'brightness': frame.bv}
        )
    
    def _compute_trust(self, frame: FrameSummary, gains: np.ndarray) -> float:
        """Compute trust score based on scene characteristics."""
        # Higher trust for moderate brightness
        brightness_factor = np.clip((frame.bv + 6) / 12.0, 0.1, 0.9)
        
        # Lower trust for extreme gains (likely incorrect)
        gain_stability = 1.0 - np.std(gains) / np.mean(gains)
        gain_stability = np.clip(gain_stability, 0.1, 1.0)
        
        # Scene-specific adjustments
        scene_factor = 0.8
        if frame.scene == "outdoor":
            scene_factor = 0.9
        elif frame.scene == "indoor":
            scene_factor = 0.7
        
        return brightness_factor * gain_stability * scene_factor

class FaceAwareAnalyzer(AWBAnalyzer):
    """
    Face-aware AWB analyzer that prioritizes skin tone accuracy.
    """
    
    def __init__(self, name: str = "face_aware"):
        super().__init__(name)
        self.skin_tone_target = 0.48  # Slightly warmer than neutral
    
    def analyze(self, frame: FrameSummary) -> AnalyzerResult:
        """Analyze focusing on face regions for skin tone accuracy."""
        if not frame.face_rois:
            # No faces detected - low trust
            return AnalyzerResult(
                gains=np.array([1.0, 1.0, 1.0], dtype=np.float32),
                cct=4000.0,
                trust=0.01,
                tag=self.name,
                metadata={'faces_detected': 0}
            )
        
        # Extract face region statistics
        face_stats = self._extract_face_stats(frame)
        if face_stats is None:
            return AnalyzerResult(
                gains=np.array([1.0, 1.0, 1.0], dtype=np.float32),
                cct=4000.0,
                trust=0.01,
                tag=self.name,
                metadata={'faces_detected': len(frame.face_rois), 'valid_faces': 0}
            )
        
        r_mean, g_mean, b_mean = face_stats
        
        # Compute gains optimized for skin tone
        gains = compute_awb_gains(r_mean, g_mean, b_mean, self.skin_tone_target)
        
        # Estimate CCT
        cct = estimate_cct_from_chromaticity(r_mean, g_mean, b_mean)
        
        # Trust based on face quality and brightness
        trust = self._compute_face_trust(frame, face_stats)
        
        return AnalyzerResult(
            gains=gains,
            cct=cct,
            trust=trust,
            tag=self.name,
            metadata={
                'faces_detected': len(frame.face_rois),
                'valid_faces': 1,
                'face_brightness': np.mean([r_mean, g_mean, b_mean])
            }
        )
    
    def _extract_face_stats(self, frame: FrameSummary) -> Optional[Tuple[float, float, float]]:
        """Extract statistics from face regions."""
        H, W = frame.r_grid.shape
        face_pixels = []
        
        for x, y, w, h in frame.face_rois:
            # Map ROI coordinates to grid coordinates
            x0 = max(0, int(x * W / frame.r_grid.shape[1]))
            y0 = max(0, int(y * H / frame.r_grid.shape[0]))
            x1 = min(W, int((x + w) * W / frame.r_grid.shape[1]))
            y1 = min(H, int((y + h) * H / frame.r_grid.shape[0]))
            
            if x1 <= x0 or y1 <= y0:
                continue
            
            # Extract region
            r_vals = frame.r_grid[y0:y1, x0:x1].flatten()
            g_vals = frame.g_grid[y0:y1, x0:x1].flatten()
            b_vals = frame.b_grid[y0:y1, x0:x1].flatten()
            
            face_pixels.extend(zip(r_vals, g_vals, b_vals))
        
        if not face_pixels:
            return None
        
        # Compute mean across all face pixels
        r_vals, g_vals, b_vals = zip(*face_pixels)
        return float(np.mean(r_vals)), float(np.mean(g_vals)), float(np.mean(b_vals))
    
    def _compute_face_trust(self, frame: FrameSummary, face_stats: Tuple[float, float, float]) -> float:
        """Compute trust based on face region quality."""
        r_mean, g_mean, b_mean = face_stats
        brightness = np.mean([r_mean, g_mean, b_mean])
        
        # Higher trust for well-lit faces
        brightness_factor = np.clip(brightness / 128.0, 0.3, 0.9)
        
        # Multiple faces increase confidence
        face_count_factor = min(len(frame.face_rois) / 3.0, 1.0)
        
        return 0.6 + 0.4 * brightness_factor * face_count_factor

class SegmentationAnalyzer(AWBAnalyzer):
    """
    Segmentation-based AWB analyzer that applies different
    corrections to different image regions.
    """
    
    def __init__(self, name: str = "segmentation"):
        super().__init__(name)
        self.segment_weights = {
            0: 0.3,  # Background
            1: 0.8,  # Person/Skin
            2: 0.6,  # Sky
            3: 0.5,  # Other objects
        }
    
    def analyze(self, frame: FrameSummary) -> AnalyzerResult:
        """Analyze using segmentation information."""
        if frame.segmentation_map is None:
            return AnalyzerResult(
                gains=np.array([1.0, 1.0, 1.0], dtype=np.float32),
                cct=4000.0,
                trust=0.05,
                tag=self.name,
                metadata={'segmentation_available': False}
            )
        
        # Analyze each segment
        segment_results = self._analyze_segments(frame)
        if not segment_results:
            return AnalyzerResult(
                gains=np.array([1.0, 1.0, 1.0], dtype=np.float32),
                cct=4000.0,
                trust=0.05,
                tag=self.name,
                metadata={'segments_found': 0}
            )
        
        # Weighted combination of segment results
        combined_result = self._combine_segment_results(segment_results)
        
        return AnalyzerResult(
            gains=combined_result['gains'],
            cct=combined_result['cct'],
            trust=combined_result['trust'],
            tag=self.name,
            metadata={
                'segments_analyzed': len(segment_results),
                'segmentation_available': True
            }
        )
    
    def _analyze_segments(self, frame: FrameSummary) -> List[Dict]:
        """Analyze individual segments."""
        unique_segments = np.unique(frame.segmentation_map)
        results = []
        
        for segment_id in unique_segments:
            mask = (frame.segmentation_map == segment_id)
            if np.sum(mask) < 10:  # Skip very small segments
                continue
            
            r_mean, g_mean, b_mean = frame.region_means(mask)
            gains = compute_awb_gains(r_mean, g_mean, b_mean)
            cct = estimate_cct_from_chromaticity(r_mean, g_mean, b_mean)
            
            weight = self.segment_weights.get(segment_id, 0.4)
            pixel_count = np.sum(mask)
            
            results.append({
                'segment_id': segment_id,
                'gains': gains,
                'cct': cct,
                'weight': weight,
                'pixel_count': pixel_count,
                'brightness': np.mean([r_mean, g_mean, b_mean])
            })
        
        return results
    
    def _combine_segment_results(self, segment_results: List[Dict]) -> Dict:
        """Combine results from multiple segments."""
        total_weight = sum(r['weight'] * r['pixel_count'] for r in segment_results)
        if total_weight == 0:
            return {'gains': np.array([1.0, 1.0, 1.0]), 'cct': 4000.0, 'trust': 0.1}
        
        # Weighted average
        combined_gains = np.zeros(3, dtype=np.float32)
        combined_cct = 0.0
        combined_trust = 0.0
        
        for result in segment_results:
            weight = result['weight'] * result['pixel_count'] / total_weight
            combined_gains += weight * result['gains']
            combined_cct += weight * result['cct']
            combined_trust += weight * 0.7  # Base trust for segments
        
        return {
            'gains': combined_gains,
            'cct': combined_cct,
            'trust': min(combined_trust, 0.9)
        }

class MLAnalyzer(AWBAnalyzer):
    """
    Machine Learning-based AWB analyzer.
    
    This is a placeholder implementation. In production, this would
    integrate with trained neural networks for AWB prediction.
    """
    
    def __init__(self, name: str = "ml"):
        super().__init__(name)
        self.low_light_threshold = 2.0
    
    def analyze(self, frame: FrameSummary) -> AnalyzerResult:
        """Analyze using ML-based approach."""
        r_mean, g_mean, b_mean = frame.global_means()
        
        # Base gains from simple heuristics
        gains = compute_awb_gains(r_mean, g_mean, b_mean)
        
        # Simulate ML behavior: better performance in low light
        if frame.bv < self.low_light_threshold:
            # Simulate ML preferring cooler tones in low light
            gains = gains * np.array([0.98, 1.0, 1.02], dtype=np.float32)
            trust = 0.7
            method = "ml_low_light"
        else:
            # Simulate ML being more conservative in good light
            gains = gains * np.array([1.0, 1.0, 1.0], dtype=np.float32)
            trust = 0.5
            method = "ml_normal_light"
        
        cct = estimate_cct_from_chromaticity(r_mean, g_mean, b_mean)
        
        return AnalyzerResult(
            gains=gains,
            cct=cct,
            trust=trust,
            tag=self.name,
            metadata={
                'method': method,
                'brightness': frame.bv,
                'is_low_light': frame.bv < self.low_light_threshold
            }
        )

# =============================================================================
# Hybrid Decision Fusion
# =============================================================================

class HybridDecisionFusion:
    """
    Fuses multiple analyzer results using trust-based weighting
    and advanced fusion strategies.
    """
    
    def __init__(self, fusion_strategy: str = "weighted_average"):
        self.fusion_strategy = fusion_strategy
    
    def fuse(self, results: List[AnalyzerResult], frame: FrameSummary) -> AnalyzerResult:
        """Fuse multiple analyzer results into a single decision."""
        if not results:
            return AnalyzerResult(
                gains=np.array([1.0, 1.0, 1.0], dtype=np.float32),
                cct=4000.0,
                trust=0.1,
                tag="fused"
            )
        
        if len(results) == 1:
            return results[0]
        
        if self.fusion_strategy == "weighted_average":
            return self._weighted_average_fusion(results, frame)
        elif self.fusion_strategy == "trust_threshold":
            return self._trust_threshold_fusion(results, frame)
        else:
            return self._weighted_average_fusion(results, frame)
    
    def _weighted_average_fusion(self, results: List[AnalyzerResult], 
                               frame: FrameSummary) -> AnalyzerResult:
        """Weighted average fusion based on trust scores."""
        # Normalize trust scores
        trusts = np.array([r.trust for r in results])
        if trusts.sum() == 0:
            trusts = np.ones(len(results)) / len(results)
        else:
            trusts = trusts / trusts.sum()
        
        # Weighted combination
        combined_gains = np.sum([w * r.gains for w, r in zip(trusts, results)], axis=0)
        combined_cct = float(np.sum([w * r.cct for w, r in zip(trusts, results)]))
        combined_trust = float(np.max(trusts))  # Use highest trust as overall confidence
        
        return AnalyzerResult(
            gains=combined_gains,
            cct=combined_cct,
            trust=combined_trust,
            tag="hybrid_weighted",
            metadata={
                'fusion_strategy': 'weighted_average',
                'analyzer_weights': trusts.tolist(),
                'analyzer_tags': [r.tag for r in results]
            }
        )
    
    def _trust_threshold_fusion(self, results: List[AnalyzerResult], 
                              frame: FrameSummary) -> AnalyzerResult:
        """Trust threshold fusion - use only high-confidence analyzers."""
        threshold = 0.3
        high_trust_results = [r for r in results if r.trust >= threshold]
        
        if not high_trust_results:
            # Fallback to best single result
            best_result = max(results, key=lambda r: r.trust)
            return AnalyzerResult(
                gains=best_result.gains,
                cct=best_result.cct,
                trust=best_result.trust * 0.8,  # Reduce trust for fallback
                tag="hybrid_threshold_fallback",
                metadata={'fusion_strategy': 'trust_threshold_fallback'}
            )
        
        # Use only high-trust results
        return self._weighted_average_fusion(high_trust_results, frame)

# =============================================================================
# Temporal Filtering and Convergence Control
# =============================================================================

class TemporalFilter:
    """
    Temporal filtering for smooth AWB transitions and convergence control.
    """
    
    def __init__(self, alpha_gains: float = 0.3, alpha_cct: float = 0.2, 
                 max_gain_change: float = 0.1):
        self.alpha_gains = alpha_gains
        self.alpha_cct = alpha_cct
        self.max_gain_change = max_gain_change
        
        # State variables
        self.prev_gains = None
        self.prev_cct = None
        self.frame_count = 0
    
    def filter(self, result: AnalyzerResult) -> AnalyzerResult:
        """Apply temporal filtering to AWB result."""
        if self.prev_gains is None:
            # First frame - no filtering
            self.prev_gains = result.gains.copy()
            self.prev_cct = result.cct
            self.frame_count = 1
            return result
        
        # Apply convergence control (limit gain changes)
        constrained_gains = self._constrain_gain_changes(result.gains)
        
        # Apply IIR filtering
        filtered_gains = (self.alpha_gains * constrained_gains + 
                         (1.0 - self.alpha_gains) * self.prev_gains)
        
        filtered_cct = (self.alpha_cct * result.cct + 
                       (1.0 - self.alpha_cct) * self.prev_cct)
        
        # Update state
        self.prev_gains = filtered_gains
        self.prev_cct = filtered_cct
        self.frame_count += 1
        
        return AnalyzerResult(
            gains=filtered_gains,
            cct=filtered_cct,
            trust=result.trust,
            tag=result.tag + "_temporal",
            metadata={
                **result.metadata,
                'temporal_filtered': True,
                'frame_count': self.frame_count
            }
        )
    
    def _constrain_gain_changes(self, new_gains: np.ndarray) -> np.ndarray:
        """Limit the maximum change in gains per frame."""
        if self.prev_gains is None:
            return new_gains
        
        delta = new_gains - self.prev_gains
        clipped_delta = np.clip(delta, -self.max_gain_change, self.max_gain_change)
        return self.prev_gains + clipped_delta
    
    def reset(self):
        """Reset temporal filter state."""
        self.prev_gains = None
        self.prev_cct = None
        self.frame_count = 0

# =============================================================================
# Color Correction Matrix Selection
# =============================================================================

class CCMSelector:
    """
    Color Correction Matrix (CCM) selection based on CCT and scene characteristics.
    """
    
    def __init__(self):
        # Predefined CCM matrices for different CCT ranges
        self.ccm_database = {
            'tungsten': np.array([
                [1.2, -0.1, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -0.1, 1.3]
            ], dtype=np.float32),
            'fluorescent': np.array([
                [1.1, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.2]
            ], dtype=np.float32),
            'daylight': np.array([
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0]
            ], dtype=np.float32),
            'cloudy': np.array([
                [0.9, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.1]
            ], dtype=np.float32),
        }
    
    def select_ccm(self, cct: float, bv: float, scene: Optional[str] = None) -> np.ndarray:
        """Select appropriate CCM based on CCT and scene."""
        # Determine illuminant type from CCT
        if cct < 3000:
            illuminant = 'tungsten'
        elif cct < 4000:
            illuminant = 'fluorescent'
        elif cct < 6500:
            illuminant = 'daylight'
        else:
            illuminant = 'cloudy'
        
        # Get base CCM
        base_ccm = self.ccm_database[illuminant].copy()
        
        # Apply scene-specific adjustments
        if scene == 'portrait':
            # Slightly warmer for portraits
            base_ccm[0, 0] *= 1.02  # Boost red
            base_ccm[2, 2] *= 0.99  # Slightly reduce blue
        
        # Apply brightness-based adjustments
        if bv < 2.0:  # Low light
            base_ccm[0, 0] *= 0.98  # Reduce red in low light
            base_ccm[2, 2] *= 1.01  # Slightly boost blue
        
        return base_ccm

# =============================================================================
# Main AWB System
# =============================================================================

class AutoWhiteBalanceSystem:
    """
    Main Auto White Balance system orchestrator.
    """
    
    def __init__(self, analyzers: Optional[List[AWBAnalyzer]] = None,
                 fusion_strategy: str = "weighted_average",
                 temporal_alpha_gains: float = 0.3,
                 temporal_alpha_cct: float = 0.2):
        
        # Initialize analyzers
        if analyzers is None:
            self.analyzers = [
                HeuristicAnalyzer(),
                FaceAwareAnalyzer(),
                SegmentationAnalyzer(),
                MLAnalyzer()
            ]
        else:
            self.analyzers = analyzers
        
        # Initialize components
        self.stats_extractor = FrameStatsExtractor()
        self.fusion = HybridDecisionFusion(fusion_strategy)
        self.temporal_filter = TemporalFilter(
            alpha_gains=temporal_alpha_gains,
            alpha_cct=temporal_alpha_cct
        )
        self.ccm_selector = CCMSelector()
        
        # Performance tracking
        self.processing_times = []
        self.frame_count = 0
    
    def process_frame(self, frame_input: Union[np.ndarray, FrameSummary],
                     **kwargs) -> AWBSystemOutput:
        """
        Process a single frame through the AWB pipeline.
        
        Args:
            frame_input: Input image (BGR) or FrameSummary object
            **kwargs: Additional frame metadata
        
        Returns:
            AWBSystemOutput with final AWB parameters
        """
        start_time = time.time()
        
        # Extract or prepare frame summary
        if isinstance(frame_input, FrameSummary):
            frame = frame_input
        else:
            frame = self.stats_extractor.extract_from_image(frame_input, **kwargs)
        
        # Run all analyzers
        analyzer_results = []
        for analyzer in self.analyzers:
            try:
                result = analyzer.analyze(frame)
                analyzer_results.append(result)
            except Exception as e:
                print(f"Warning: Analyzer {analyzer.name} failed: {e}")
                continue
        
        # Fuse analyzer results
        if analyzer_results:
            fused_result = self.fusion.fuse(analyzer_results, frame)
        else:
            # Fallback if all analyzers fail
            fused_result = AnalyzerResult(
                gains=np.array([1.0, 1.0, 1.0], dtype=np.float32),
                cct=4000.0,
                trust=0.1,
                tag="fallback"
            )
        
        # Apply temporal filtering
        filtered_result = self.temporal_filter.filter(fused_result)
        
        # Select CCM
        ccm = self.ccm_selector.select_ccm(
            filtered_result.cct, frame.bv, frame.scene
        )
        
        # Apply user preferences (if any)
        final_gains = self._apply_preferences(filtered_result.gains, frame)
        
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Update performance tracking
        self.processing_times.append(processing_time)
        self.frame_count += 1
        
        return AWBSystemOutput(
            gains=final_gains,
            cct=filtered_result.cct,
            ccm=ccm,
            trust=filtered_result.trust,
            analyzer_results=analyzer_results,
            processing_time=processing_time,
            metadata={
                'frame_count': self.frame_count,
                'scene': frame.scene,
                'brightness_value': frame.bv,
                'fusion_strategy': self.fusion.fusion_strategy
            }
        )
    
    def _apply_preferences(self, gains: np.ndarray, frame: FrameSummary) -> np.ndarray:
        """Apply scene-specific or user preferences."""
        adjusted_gains = gains.copy()
        
        # Scene-specific adjustments
        if frame.scene == 'portrait':
            # Slightly warmer for portraits
            adjusted_gains *= np.array([1.02, 1.0, 0.99], dtype=np.float32)
        elif frame.scene == 'landscape':
            # Slightly cooler for landscapes
            adjusted_gains *= np.array([0.99, 1.0, 1.01], dtype=np.float32)
        
        return adjusted_gains
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics."""
        if not self.processing_times:
            return {'average_time': 0, 'max_time': 0, 'min_time': 0, 'frame_count': 0}
        
        return {
            'average_time': np.mean(self.processing_times),
            'max_time': np.max(self.processing_times),
            'min_time': np.min(self.processing_times),
            'frame_count': self.frame_count,
            'fps_estimate': 1000.0 / np.mean(self.processing_times) if self.processing_times else 0
        }
    
    def reset_temporal_state(self):
        """Reset temporal filter state."""
        self.temporal_filter.reset()
        self.frame_count = 0
        self.processing_times = []

# =============================================================================
# Example Usage and Testing
# =============================================================================

def create_test_image(width: int = 640, height: int = 480, 
                     color_temperature: str = "warm") -> np.ndarray:
    """Create a test image with specific color characteristics."""
    # Create base image
    image = np.ones((height, width, 3), dtype=np.float32) * 128
    
    # Add color bias based on temperature
    if color_temperature == "warm":
        image[:, :, 2] *= 1.3  # Boost red
        image[:, :, 0] *= 0.8  # Reduce blue
    elif color_temperature == "cool":
        image[:, :, 0] *= 1.3  # Boost blue
        image[:, :, 2] *= 0.8  # Reduce red
    
    # Add some noise
    noise = np.random.normal(0, 10, image.shape)
    image = np.clip(image + noise, 0, 255)
    
    return image.astype(np.uint8)

def main():
    """Example usage of the AWB system."""
    print("Auto White Balance System Demo")
    print("=" * 40)
    
    # Create test images
    warm_image = create_test_image(color_temperature="warm")
    cool_image = create_test_image(color_temperature="cool")
    
    # Initialize AWB system
    awb_system = AutoWhiteBalanceSystem()
    
    # Process warm image
    print("\nProcessing warm image...")
    warm_result = awb_system.process_frame(
        warm_image,
        bv=5.0,
        scene="portrait",
        face_rois=[(100, 100, 80, 80)]  # Simulate face region
    )
    
    print(f"Warm image results:")
    print(f"  Gains: {warm_result.gains}")
    print(f"  CCT: {warm_result.cct:.1f}K")
    print(f"  Trust: {warm_result.trust:.3f}")
    print(f"  Processing time: {warm_result.processing_time:.2f}ms")
    
    # Process cool image
    print("\nProcessing cool image...")
    cool_result = awb_system.process_frame(
        cool_image,
        bv=7.0,
        scene="landscape"
    )
    
    print(f"Cool image results:")
    print(f"  Gains: {cool_result.gains}")
    print(f"  CCT: {cool_result.cct:.1f}K")
    print(f"  Trust: {cool_result.trust:.3f}")
    print(f"  Processing time: {cool_result.processing_time:.2f}ms")
    
    # Performance statistics
    stats = awb_system.get_performance_stats()
    print(f"\nPerformance Statistics:")
    print(f"  Average processing time: {stats['average_time']:.2f}ms")
    print(f"  Estimated FPS: {stats['fps_estimate']:.1f}")
    print(f"  Frames processed: {stats['frame_count']}")

if __name__ == "__main__":
    main()
