"""
Advanced Auto White Balance (AWB) System

A comprehensive AWB implementation inspired by hybrid approaches, featuring:
- Multi-analyzer framework (heuristic, ML, face-aware, segmentation)
- Trust-based fusion with temporal smoothing
- CCM selection and spatial color correction
- Real-time processing optimization

This module implements a production-ready Auto White Balance system that:
1. Extracts color statistics from images or sensor data
2. Runs multiple AWB analyzers in parallel
3. Fuses analyzer results using trust-based weighting
4. Applies temporal filtering for smooth video transitions
5. Selects appropriate Color Correction Matrix (CCM) based on scene

Key Concepts:
- Gray World Assumption: Average scene color should be neutral gray
- Color Temperature (CCT): Describes the color of light (warm to cool)
- AWB Gains: Multipliers applied to RGB channels to correct color cast
- Trust Scores: Confidence levels for each analyzer's decision
- Temporal Filtering: Smooths AWB decisions across video frames

Usage Example:
    from src.awb_system import AutoWhiteBalanceSystem
    import cv2
    
    # Initialize system
    awb = AutoWhiteBalanceSystem()
    
    # Process image
    image = cv2.imread('photo.jpg')
    result = awb.process_frame(image, bv=5.0, scene="portrait")
    
    # Apply correction
    corrected = image.copy()
    corrected[:, :, 0] *= result.gains[2]  # Blue
    corrected[:, :, 1] *= result.gains[1]  # Green
    corrected[:, :, 2] *= result.gains[0]  # Red

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
# These functions provide basic color processing operations used throughout
# the AWB system. They handle safe numerical operations and color space
# conversions.

def safe_divide(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Safe division with epsilon to avoid division by zero.
    
    This is critical for AWB gain calculations where we divide by channel
    means that could be zero or very small values.
    
    Args:
        a: Numerator array
        b: Denominator array
        eps: Small epsilon value to add to denominator (default: 1e-8)
    
    Returns:
        Division result: a / (b + eps)
    
    Example:
        # Without safe_divide, this would cause division by zero:
        gains = mean_values / channel_means  # Error if channel_means = 0
        
        # With safe_divide:
        gains = safe_divide(mean_values, channel_means)  # Safe!
    """
    return a / (b + eps)

def rgb_to_chromaticity(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert RGB values to chromaticity coordinates (normalized color).
    
    Chromaticity removes brightness information, leaving only color information.
    This is useful for color temperature estimation independent of scene brightness.
    
    Formula: x = R/(R+G+B), y = G/(R+G+B), z = B/(R+G+B)
    Note: x + y + z = 1.0 (only two values are independent)
    
    Args:
        r: Red channel values
        g: Green channel values
        b: Blue channel values
    
    Returns:
        Tuple of (x, y, z) chromaticity coordinates
    
    Example:
        # Warm light (more red/yellow)
        r, g, b = 200, 150, 100
        x, y, z = rgb_to_chromaticity(r, g, b)
        # x will be higher (more red), z will be lower (less blue)
    """
    s = r + g + b + 1e-12  # Sum with epsilon to avoid division by zero
    return r / s, g / s, b / s

def estimate_cct_from_chromaticity(r: float, g: float, b: float) -> float:
    """
    Estimate Correlated Color Temperature (CCT) from RGB chromaticity.
    
    Color Temperature describes the color of light:
    - Low CCT (2000-3000K): Warm light (orange/yellow) - incandescent bulbs
    - Medium CCT (4000-5000K): Neutral light (white) - fluorescent, daylight
    - High CCT (6000-10000K): Cool light (blue) - shade, overcast sky
    
    This is a simplified estimation based on blue ratio. For production use,
    implement proper Planckian locus mapping or use trained ML models.
    
    Algorithm:
        1. Compute blue ratio: blue / (red + green + blue)
        2. Map to CCT range: 2000K (warm) to 10000K (cool)
        3. More blue = higher CCT (cooler light)
    
    Args:
        r: Red channel mean value
        g: Green channel mean value
        b: Blue channel mean value
    
    Returns:
        Estimated CCT in Kelvin (range: 2000-10000K)
    
    Example:
        # Warm scene (more red, less blue)
        cct = estimate_cct_from_chromaticity(200, 150, 100)
        # Returns ~3500K (warm)
        
        # Cool scene (less red, more blue)
        cct = estimate_cct_from_chromaticity(100, 150, 200)
        # Returns ~6500K (cool)
    """
    # Convert to chromaticity (normalized color)
    x, y, _ = rgb_to_chromaticity(r, g, b)
    
    # Simple heuristic: blue ratio indicates color temperature
    # More blue = higher CCT (cooler light)
    blue_ratio = b / (r + g + b + 1e-8)
    
    # Map blue ratio [0, 1] to CCT [2000K, 10000K]
    # 0 = all red/green (warm) -> 2000K
    # 1 = all blue (cool) -> 10000K
    return 2000 + 8000 * np.clip(blue_ratio, 0, 1)

def compute_awb_gains(r_mean: float, g_mean: float, b_mean: float, 
                     target_g: float = 0.5) -> np.ndarray:
    """
    Compute AWB gains using Gray World Assumption.
    
    Gray World Assumption: The average color of a scene should be neutral gray.
    This function calculates multipliers (gains) for each RGB channel to achieve
    this neutral gray balance.
    
    Algorithm:
        1. Calculate desired green value: total * target_g
        2. Compute gains to make each channel equal to desired green
        3. Normalize so green gain = 1.0 (green is reference channel)
    
    Mathematical Derivation:
        Original: (R_mean, G_mean, B_mean)
        After correction: (R_mean * R_gain, G_mean * G_gain, B_mean * B_gain)
        Goal: R_mean * R_gain = G_mean * G_gain = B_mean * B_gain = desired_g
        
        Solving:
            R_gain = desired_g / R_mean
            G_gain = desired_g / G_mean
            B_gain = desired_g / B_mean
        
        Normalize to G_gain = 1.0:
            R_gain = (desired_g / R_mean) / (desired_g / G_mean) = G_mean / R_mean
            B_gain = (desired_g / B_mean) / (desired_g / G_mean) = G_mean / B_mean
    
    Args:
        r_mean: Mean red channel value
        g_mean: Mean green channel value (reference channel)
        b_mean: Mean blue channel value
        target_g: Target green ratio (0.0-1.0). Default 0.5 means green should
                 be 50% of total intensity (neutral gray)
    
    Returns:
        Array of [R_gain, G_gain, B_gain] where G_gain = 1.0
    
    Example:
        # Warm scene: too much red, not enough blue
        r, g, b = 200, 150, 100
        gains = compute_awb_gains(r, g, b)
        # Returns: [0.75, 1.0, 1.5]
        # Interpretation:
        #   - Reduce red by 25% (0.75)
        #   - Keep green unchanged (1.0)
        #   - Boost blue by 50% (1.5)
        # Result: Balanced colors
        
        # After applying gains:
        # R_new = 200 * 0.75 = 150
        # G_new = 150 * 1.0 = 150
        # B_new = 100 * 1.5 = 150
        # All channels now equal = neutral gray!
    """
    # Calculate total intensity
    total = r_mean + g_mean + b_mean + 1e-12
    
    # Desired green value based on target ratio
    # For gray world (target_g=0.5), green should be 50% of total
    desired_g = total * target_g
    
    # Calculate gains to normalize each channel to desired_g
    g_gain = safe_divide(desired_g, g_mean)
    r_gain = safe_divide(desired_g, r_mean)
    b_gain = safe_divide(desired_g, b_mean)
    
    # Normalize so G gain = 1.0 (green is the reference channel)
    # This is standard practice: green is most sensitive and stable
    r_gain /= g_gain
    b_gain /= g_gain
    g_gain = 1.0
    
    return np.array([r_gain, g_gain, b_gain], dtype=np.float32)

# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class FrameSummary:
    """
    Container for frame statistics and metadata.
    
    This class holds all the information needed for AWB processing:
    - Color statistics (RGB grids) - subsampled for efficiency
    - Scene metadata (brightness, scene type, faces, etc.)
    - Optional advanced data (segmentation, spectral, depth)
    
    The RGB grids are typically 64x48 (or similar small size) to enable
    real-time processing without analyzing full-resolution images.
    
    Attributes:
        r_grid: Red channel statistics grid (H x W, typically 48x64)
                Each pixel represents average red value in that region
        g_grid: Green channel statistics grid (same size as r_grid)
        b_grid: Blue channel statistics grid (same size as r_grid)
        bv: Brightness Value from Auto Exposure system
            Range: -6 (very dark) to +15 (very bright)
            Typical: 2-10 for normal scenes
        scene: Scene type identifier (e.g., "portrait", "landscape", "indoor")
               Used for scene-specific AWB adjustments
        face_rois: List of face regions as (x, y, width, height) tuples
                   Coordinates are in original image space
                   Used by FaceAwareAnalyzer for skin tone optimization
        segmentation_map: Semantic segmentation map (same size as grids)
                          Integer values: 0=background, 1=person, 2=sky, etc.
                          Used by SegmentationAnalyzer for region-based AWB
        spectral_data: Optional spectral sensor data (illuminant type, etc.)
        depth_info: Optional depth information from AF system
        timestamp: Frame timestamp for temporal processing
    
    Example:
        # Create from image
        extractor = FrameStatsExtractor()
        frame = extractor.extract_from_image(
            image,
            bv=5.0,
            scene="portrait",
            face_rois=[(100, 100, 80, 80)]
        )
        
        # Access statistics
        r_mean, g_mean, b_mean = frame.global_means()
    """
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
    """
    Result from a single AWB analyzer.
    
    Each analyzer (Heuristic, Face-aware, ML, etc.) produces an AnalyzerResult
    containing its AWB decision and confidence level. These results are then
    fused together by HybridDecisionFusion.
    
    Attributes:
        gains: AWB correction gains as [R_gain, G_gain, B_gain]
               These are multipliers applied to RGB channels
               Example: [1.2, 1.0, 0.9] means:
                 - Boost red by 20%
                 - Keep green unchanged (reference)
                 - Reduce blue by 10%
        cct: Estimated Color Temperature in Kelvin
              Range: 2000K (warm) to 10000K (cool)
              Typical: 3000K (incandescent) to 6500K (daylight)
        trust: Confidence score in the analyzer's decision [0.0, 1.0]
               0.0 = no confidence, 1.0 = high confidence
               Used for weighted fusion of multiple analyzers
        tag: Analyzer identifier (e.g., "heuristic", "face_aware", "ml")
        metadata: Additional information (method used, brightness, etc.)
    
    Example:
        # Heuristic analyzer result
        result = AnalyzerResult(
            gains=np.array([1.1, 1.0, 0.95]),
            cct=4500.0,
            trust=0.75,
            tag="heuristic",
            metadata={'method': 'gray_world', 'brightness': 6.0}
        )
        
        # Low trust result (analyzer uncertain)
        result = AnalyzerResult(
            gains=np.array([1.0, 1.0, 1.0]),
            cct=4000.0,
            trust=0.1,
            tag="face_aware",
            metadata={'faces_detected': 0}  # No faces = low trust
        )
    """
    gains: np.ndarray    # AWB gains [R, G, B]
    cct: float          # Estimated CCT
    trust: float        # Confidence score [0, 1]
    tag: str = ""       # Analyzer identifier
    metadata: Dict = None  # Additional metadata
    
    def __post_init__(self):
        """Initialize metadata dictionary if not provided."""
        if self.metadata is None:
            self.metadata = {}

@dataclass
class AWBSystemOutput:
    """
    Final output from the complete AWB system pipeline.
    
    This contains the final AWB decision after:
    1. Running all analyzers
    2. Fusing their results
    3. Applying temporal filtering
    4. Selecting CCM
    
    Attributes:
        gains: Final AWB gains [R, G, B] ready to apply to image
               These are the result of fusion and temporal filtering
        cct: Final estimated Color Temperature (after temporal smoothing)
        ccm: Color Correction Matrix (3x3) for color space transformation
             Maps from camera RGB to standard color space (e.g., sRGB)
        trust: Overall confidence in the AWB decision [0.0, 1.0]
               Typically the maximum trust from individual analyzers
        analyzer_results: List of all individual analyzer results
                         Useful for debugging and understanding contributions
        processing_time: Time taken to process this frame (milliseconds)
                        Useful for performance monitoring
        metadata: Additional information (frame count, scene, fusion strategy)
    
    Example:
        # Process frame and get output
        result = awb_system.process_frame(image, bv=5.0, scene="portrait")
        
        # Apply gains to image
        corrected = image.copy()
        corrected[:, :, 0] *= result.gains[2]  # Blue
        corrected[:, :, 1] *= result.gains[1]  # Green
        corrected[:, :, 2] *= result.gains[0]  # Red
        
        # Check individual analyzer contributions
        for analyzer_result in result.analyzer_results:
            print(f"{analyzer_result.tag}: gains={analyzer_result.gains}, "
                  f"trust={analyzer_result.trust}")
        
        # Serialize for storage
        result_dict = result.to_dict()
    """
    gains: np.ndarray           # Final AWB gains
    cct: float                 # Estimated CCT
    ccm: np.ndarray           # Color Correction Matrix
    trust: float              # Overall confidence
    analyzer_results: List[AnalyzerResult]  # Individual analyzer results
    processing_time: float    # Processing time in ms
    metadata: Dict           # Additional metadata
    
    def to_dict(self) -> Dict:
        """
        Convert to dictionary for JSON serialization.
        
        Useful for saving results to file or sending over network.
        
        Returns:
            Dictionary representation of the output
        """
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
    """
    Extract color statistics from images or sensor data.
    
    This class efficiently extracts RGB statistics from full-resolution images
    by downsampling to a small grid (e.g., 64x48). This enables real-time
    processing without analyzing millions of pixels.
    
    The extraction process:
    1. Resize image to grid size (e.g., 4032x3024 -> 64x48)
    2. Split into RGB channels
    3. Each grid cell contains average color for that region
    
    This approach is standard in camera systems where sensor provides
    subsampled statistics directly, avoiding full image processing.
    
    Attributes:
        grid_width: Width of statistics grid (default: 64)
        grid_height: Height of statistics grid (default: 48)
    
    Example:
        # Extract from full image
        extractor = FrameStatsExtractor(grid_width=64, grid_height=48)
        frame = extractor.extract_from_image(
            image,
            bv=5.0,
            scene="portrait"
        )
        
        # Extract from precomputed Bayer statistics (more efficient)
        frame = extractor.extract_from_bayer_stats(
            r_stats, g_stats, b_stats,
            bv=5.0
        )
    """
    
    def __init__(self, grid_width: int = 64, grid_height: int = 48):
        """
        Initialize statistics extractor.
        
        Args:
            grid_width: Width of statistics grid (typically 32-128)
            grid_height: Height of statistics grid (typically 24-96)
        
        Note: Smaller grids = faster processing but less spatial detail
              Larger grids = more detail but slower processing
        """
        self.grid_width = grid_width
        self.grid_height = grid_height
    
    def extract_from_image(self, image: np.ndarray, **kwargs) -> FrameSummary:
        """
        Extract statistics from a full-resolution image.
        
        This method downsamples the image to a small grid and extracts RGB
        statistics. This is efficient for real-time processing.
        
        Process:
            1. Resize image to grid size (e.g., 4032x3024 -> 64x48)
            2. Convert BGR to RGB (OpenCV uses BGR format)
            3. Each grid cell contains average color for that region
            4. Package with metadata into FrameSummary
        
        Args:
            image: Input image in BGR format (H x W x 3, uint8 or float)
            **kwargs: Additional metadata:
                - bv: Brightness value from auto-exposure
                - scene: Scene type ("portrait", "landscape", etc.)
                - face_rois: List of face regions [(x, y, w, h), ...]
                - segmentation_map: Segmentation map (same size as grid)
                - spectral_data: Spectral sensor data
                - depth_info: Depth information
                - timestamp: Frame timestamp
        
        Returns:
            FrameSummary with extracted statistics and metadata
        
        Example:
            extractor = FrameStatsExtractor()
            frame = extractor.extract_from_image(
                image,
                bv=5.0,
                scene="portrait",
                face_rois=[(100, 100, 80, 80)]
            )
        """
        # Resize to grid size for efficient processing
        # INTER_AREA interpolation is best for downsampling (preserves averages)
        small_image = cv2.resize(
            image, (self.grid_width, self.grid_height), 
            interpolation=cv2.INTER_AREA
        ).astype(np.float32)
        
        # Split channels (OpenCV uses BGR format, we need RGB)
        # Channel order: [Blue, Green, Red] -> [Red, Green, Blue]
        b_channel = small_image[..., 0]  # Blue in BGR = index 0
        g_channel = small_image[..., 1]  # Green in BGR = index 1
        r_channel = small_image[..., 2]  # Red in BGR = index 2
        
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
    """
    Base class for all AWB analyzers.
    
    This abstract base class defines the interface that all AWB analyzers
    must implement. Each analyzer provides an independent AWB decision
    based on different algorithms or assumptions.
    
    The system uses multiple analyzers in parallel and fuses their results
    using trust-based weighting. This hybrid approach combines the strengths
    of different algorithms.
    
    Subclasses must implement:
        - analyze(): Process frame and return AnalyzerResult
    
    Example:
        class CustomAnalyzer(AWBAnalyzer):
            def analyze(self, frame):
                # Your AWB algorithm here
                gains = compute_awb_gains(...)
                return AnalyzerResult(gains, cct, trust, tag="custom")
    """
    
    def __init__(self, name: str):
        """
        Initialize analyzer with a name identifier.
        
        Args:
            name: Unique identifier for this analyzer (e.g., "heuristic", "face_aware")
        """
        self.name = name
    
    @abstractmethod
    def analyze(self, frame: FrameSummary) -> AnalyzerResult:
        """
        Analyze frame and return AWB decision.
        
        This is the main method that each analyzer must implement.
        It processes the frame statistics and returns:
        - AWB gains (RGB correction multipliers)
        - Estimated CCT (color temperature)
        - Trust score (confidence in the decision)
        
        Args:
            frame: FrameSummary containing image statistics and metadata
        
        Returns:
            AnalyzerResult with the analyzer's AWB decision
        
        Note: Each analyzer can use different algorithms:
            - HeuristicAnalyzer: Gray world assumption
            - FaceAwareAnalyzer: Skin tone optimization
            - SegmentationAnalyzer: Region-based correction
            - MLAnalyzer: Machine learning prediction
        """
        pass
    
    def __str__(self) -> str:
        """String representation for debugging."""
        return f"{self.__class__.__name__}({self.name})"

# =============================================================================
# Analyzer Implementations
# =============================================================================

class HeuristicAnalyzer(AWBAnalyzer):
    """
    Heuristic-based AWB analyzer using Gray World Assumption.
    
    This is the most fundamental AWB algorithm, based on the assumption that
    the average color of a scene should be neutral gray. This works well for
    many natural scenes with diverse colors.
    
    Algorithm:
        1. Compute global RGB means from the entire image
        2. Calculate gains to normalize channels to gray
        3. Estimate CCT from chromaticity
        4. Compute trust based on scene characteristics
    
    Strengths:
        - Simple and fast
        - Works well for diverse scenes
        - No additional data required (no faces, segmentation, etc.)
    
    Weaknesses:
        - Fails with dominant colors (e.g., green grass, blue sky)
        - Assumes scene has diverse colors
        - May over-correct scenes with intentional color casts
    
    Typical Use Cases:
        - General photography
        - Scenes with diverse colors
        - Fallback when other analyzers fail
    
    Example:
        analyzer = HeuristicAnalyzer()
        result = analyzer.analyze(frame)
        # result.gains: [R_gain, G_gain, B_gain]
        # result.trust: 0.0-1.0 (confidence)
    """
    
    def __init__(self, name: str = "heuristic"):
        """
        Initialize heuristic analyzer.
        
        Args:
            name: Analyzer identifier (default: "heuristic")
        """
        super().__init__(name)
        # Target green ratio for gray world assumption
        # 0.5 means green should be 50% of total (neutral gray)
        self.gray_world_target = 0.5
    
    def analyze(self, frame: FrameSummary) -> AnalyzerResult:
        """
        Analyze frame using Gray World Assumption.
        
        Process:
            1. Compute global RGB means (average color of entire scene)
            2. Calculate gains to normalize to gray world target
            3. Estimate color temperature from chromaticity
            4. Compute trust score based on scene quality
        
        The gray world assumption states: "The average reflectance in a scene
        is achromatic (gray)". This means if we average all colors in a scene,
        we should get neutral gray. If not, we correct for the color cast.
        
        Args:
            frame: FrameSummary with image statistics
        
        Returns:
            AnalyzerResult with:
                - gains: RGB correction multipliers
                - cct: Estimated color temperature
                - trust: Confidence score (higher for well-lit, diverse scenes)
        """
        # Step 1: Get global color statistics
        # This is the average color across the entire image
        r_mean, g_mean, b_mean = frame.global_means()
        
        # Step 2: Calculate AWB gains using gray world assumption
        # This computes multipliers to make the average color neutral gray
        gains = compute_awb_gains(r_mean, g_mean, b_mean, self.gray_world_target)
        
        # Step 3: Estimate Color Temperature
        # This helps understand the lighting conditions
        cct = estimate_cct_from_chromaticity(r_mean, g_mean, b_mean)
        
        # Step 4: Compute trust score
        # Higher trust for well-lit scenes with stable gains
        trust = self._compute_trust(frame, gains)
        
        return AnalyzerResult(
            gains=gains,
            cct=cct,
            trust=trust,
            tag=self.name,
            metadata={'method': 'gray_world', 'brightness': frame.bv}
        )
    
    def _compute_trust(self, frame: FrameSummary, gains: np.ndarray) -> float:
        """
        Compute trust score based on scene characteristics.
        
        Trust score indicates confidence in the AWB decision. Higher trust
        means the gray world assumption is more likely to be valid.
        
        Factors considered:
        1. Brightness: Well-lit scenes are more reliable
        2. Gain stability: Extreme gains suggest unusual scene (low trust)
        3. Scene type: Outdoor scenes typically more diverse (higher trust)
        
        Trust Formula:
            trust = brightness_factor × gain_stability × scene_factor
        
        Args:
            frame: FrameSummary with scene metadata
            gains: Computed AWB gains [R, G, B]
        
        Returns:
            Trust score [0.0, 1.0] where:
            - > 0.7: High confidence (well-lit, diverse scene)
            - 0.4-0.7: Moderate confidence (reasonable scene)
            - < 0.4: Low confidence (unusual scene, may need manual adjustment)
        
        Example:
            # Well-lit outdoor scene
            trust = _compute_trust(frame, gains=[1.05, 1.0, 0.98])
            # Returns ~0.8 (high trust)
            
            # Dark scene with extreme gains
            trust = _compute_trust(frame, gains=[2.0, 1.0, 0.5])
            # Returns ~0.3 (low trust - gains too extreme)
        """
        # Factor 1: Brightness factor
        # BV range: -6 (very dark) to +15 (very bright)
        # Optimal: BV 2-10 (moderate to bright)
        # Formula maps BV to [0.1, 0.9] range
        brightness_factor = np.clip((frame.bv + 6) / 12.0, 0.1, 0.9)
        
        # Factor 2: Gain stability
        # If gains are very different from each other, scene is unusual
        # Example: [2.0, 1.0, 0.5] = high std = low stability = low trust
        # Example: [1.05, 1.0, 0.98] = low std = high stability = high trust
        gain_std = np.std(gains)
        gain_mean = np.mean(gains)
        gain_stability = 1.0 - (gain_std / (gain_mean + 1e-8))
        gain_stability = np.clip(gain_stability, 0.1, 1.0)
        
        # Factor 3: Scene-specific adjustments
        # Outdoor scenes typically have more diverse colors (higher trust)
        # Indoor scenes may have dominant colors (lower trust)
        scene_factor = 0.8  # Default (unknown scene)
        if frame.scene == "outdoor":
            scene_factor = 0.9  # Outdoor = more diverse = higher trust
        elif frame.scene == "indoor":
            scene_factor = 0.7  # Indoor = may have color casts = lower trust
        
        # Combined trust score
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
    Fuses multiple analyzer results using trust-based weighting.
    
    This class combines AWB decisions from multiple analyzers (heuristic,
    face-aware, ML, segmentation) into a single final decision. The fusion
    is weighted by each analyzer's trust score, giving more weight to
    high-confidence analyzers.
    
    Fusion Strategies:
        1. "weighted_average": All analyzers weighted by trust (default)
        2. "trust_threshold": Only high-confidence analyzers (trust >= 0.3)
    
    Algorithm (weighted_average):
        1. Normalize trust scores to sum to 1.0
        2. Weighted average of gains: Σ(trust[i] × gains[i])
        3. Weighted average of CCT: Σ(trust[i] × CCT[i])
        4. Final trust = max(individual trusts)
    
    Example:
        # Multiple analyzers produce different results
        heuristic_result = AnalyzerResult(gains=[1.1, 1.0, 0.9], trust=0.8)
        face_result = AnalyzerResult(gains=[1.05, 1.0, 0.95], trust=0.6)
        ml_result = AnalyzerResult(gains=[1.08, 1.0, 0.92], trust=0.5)
        
        # Fuse them
        fusion = HybridDecisionFusion()
        fused = fusion.fuse([heuristic_result, face_result, ml_result], frame)
        # Result: Weighted combination favoring heuristic (highest trust)
    """
    
    def __init__(self, fusion_strategy: str = "weighted_average"):
        """
        Initialize fusion module.
        
        Args:
            fusion_strategy: Fusion method
                - "weighted_average": Weight all analyzers by trust (default)
                - "trust_threshold": Only use analyzers with trust >= 0.3
        """
        self.fusion_strategy = fusion_strategy
    
    def fuse(self, results: List[AnalyzerResult], frame: FrameSummary) -> AnalyzerResult:
        """
        Fuse multiple analyzer results into a single decision.
        
        This is the core fusion algorithm that combines multiple AWB decisions.
        The method used depends on fusion_strategy.
        
        Args:
            results: List of AnalyzerResult from different analyzers
            frame: FrameSummary (used for context in some strategies)
        
        Returns:
            Fused AnalyzerResult with combined gains and CCT
        
        Note: If no results provided, returns neutral gains [1.0, 1.0, 1.0]
        """
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
        """
        Weighted average fusion based on trust scores.
        
        This is the default fusion strategy that combines all analyzers,
        weighting each by its trust score. Analyzers with higher trust
        contribute more to the final decision.
        
        Mathematical Formula:
            weights[i] = trust[i] / Σ(trust[j])
            final_gains = Σ(weights[i] × gains[i])
            final_cct = Σ(weights[i] × CCT[i])
            final_trust = max(trust[i])
        
        Example:
            Analyzer 1: gains=[1.1, 1.0, 0.9], trust=0.8 → weight=0.57
            Analyzer 2: gains=[1.05, 1.0, 0.95], trust=0.4 → weight=0.29
            Analyzer 3: gains=[1.08, 1.0, 0.92], trust=0.2 → weight=0.14
            
            Final gains = 0.57×[1.1,1.0,0.9] + 0.29×[1.05,1.0,0.95] + 0.14×[1.08,1.0,0.92]
                       = [1.08, 1.0, 0.91]  (closer to high-trust analyzer)
        
        Args:
            results: List of AnalyzerResult objects
            frame: FrameSummary (not used in this strategy, kept for interface)
        
        Returns:
            Fused AnalyzerResult with weighted combination
        """
        # Step 1: Extract trust scores
        trusts = np.array([r.trust for r in results])
        
        # Step 2: Normalize trust scores to sum to 1.0
        # This creates weights that sum to 1.0 for proper weighted average
        if trusts.sum() == 0:
            # Fallback: equal weights if all trusts are zero
            trusts = np.ones(len(results)) / len(results)
        else:
            # Normalize: each trust becomes a weight
            trusts = trusts / trusts.sum()
        
        # Step 3: Weighted combination of gains
        # Formula: Σ(weight[i] × gains[i]) for each channel
        combined_gains = np.sum([w * r.gains for w, r in zip(trusts, results)], axis=0)
        
        # Step 4: Weighted combination of CCT
        combined_cct = float(np.sum([w * r.cct for w, r in zip(trusts, results)]))
        
        # Step 5: Final trust = maximum individual trust
        # This indicates the highest confidence source
        combined_trust = float(np.max(trusts))
        
        return AnalyzerResult(
            gains=combined_gains,
            cct=combined_cct,
            trust=combined_trust,
            tag="hybrid_weighted",
            metadata={
                'fusion_strategy': 'weighted_average',
                'analyzer_weights': trusts.tolist(),  # For debugging
                'analyzer_tags': [r.tag for r in results]  # Which analyzers contributed
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
    Temporal filtering for smooth AWB transitions in video.
    
    This class prevents color flickering in video by smoothing AWB decisions
    across consecutive frames. It uses an IIR (Infinite Impulse Response) filter
    to gradually transition between AWB decisions.
    
    Why Temporal Filtering?
        Without filtering, AWB can "jump" between frames causing:
        - Color flickering (annoying to viewers)
        - Unstable video appearance
        - Poor user experience
    
    How It Works:
        1. Convergence Control: Limits maximum gain change per frame
        2. IIR Filtering: Smooths gains and CCT over time
        3. Formula: filtered[n] = α × current[n] + (1-α) × filtered[n-1]
    
    Parameters:
        - alpha: Smoothing strength (0.0-1.0)
          Lower α = smoother (slower response)
          Higher α = faster response (less smooth)
        - max_gain_change: Maximum allowed gain change per frame
    
    Typical Settings:
        - Photo mode: α=0.5 (faster convergence, less smoothing)
        - Video mode: α=0.2-0.3 (smoother, prevent flicker)
        - Low light: α=0.2 (prevent flicker in challenging conditions)
    
    Example:
        filter = TemporalFilter(alpha_gains=0.3, alpha_cct=0.2)
        
        # Frame 1: gains=[1.0, 1.0, 1.0]
        result1 = filter.filter(AnalyzerResult(gains=[1.0, 1.0, 1.0], ...))
        # Output: [1.0, 1.0, 1.0] (first frame, no filtering)
        
        # Frame 2: gains=[1.2, 1.0, 0.9] (large change)
        result2 = filter.filter(AnalyzerResult(gains=[1.2, 1.0, 0.9], ...))
        # Output: [1.06, 1.0, 0.97] (smoothed: 0.3×[1.2,1.0,0.9] + 0.7×[1.0,1.0,1.0])
    """
    
    def __init__(self, alpha_gains: float = 0.3, alpha_cct: float = 0.2, 
                 max_gain_change: float = 0.1):
        """
        Initialize temporal filter.
        
        Args:
            alpha_gains: Smoothing factor for gains (0.1-0.5 recommended)
                        Lower = smoother, Higher = faster response
            alpha_cct: Smoothing factor for CCT (typically lower than alpha_gains)
                      CCT changes are usually smoother than gain changes
            max_gain_change: Maximum allowed gain change per frame (0.05-0.2)
                           Prevents sudden jumps, ensures convergence control
        """
        self.alpha_gains = alpha_gains
        self.alpha_cct = alpha_cct
        self.max_gain_change = max_gain_change
        
        # State variables (maintain filter state across frames)
        self.prev_gains = None  # Previous frame's filtered gains
        self.prev_cct = None    # Previous frame's filtered CCT
        self.frame_count = 0    # Frame counter for debugging
    
    def filter(self, result: AnalyzerResult) -> AnalyzerResult:
        """
        Apply temporal filtering to AWB result.
        
        Process:
            1. First frame: No filtering (initialize state)
            2. Convergence control: Limit gain changes
            3. IIR filtering: Smooth transition from previous frame
            4. Update state for next frame
        
        Args:
            result: Current frame's AnalyzerResult from fusion
        
        Returns:
            Temporally filtered AnalyzerResult
        """
        # First frame: Initialize state (no filtering needed)
        if self.prev_gains is None:
            self.prev_gains = result.gains.copy()
            self.prev_cct = result.cct
            self.frame_count = 1
            return result  # Return as-is for first frame
        
        # Step 1: Convergence Control
        # Limit the maximum change in gains per frame to prevent sudden jumps
        # Example: If prev_gains=[1.0,1.0,1.0] and new_gains=[1.5,1.0,0.5],
        #          and max_gain_change=0.1, then constrained=[1.1,1.0,0.9]
        constrained_gains = self._constrain_gain_changes(result.gains)
        
        # Step 2: IIR Filtering for Gains
        # Formula: filtered = α × current + (1-α) × previous
        # This creates exponential smoothing
        # Example with α=0.3:
        #   filtered = 0.3 × [1.1,1.0,0.9] + 0.7 × [1.0,1.0,1.0]
        #            = [1.03, 1.0, 0.97]  (smooth transition)
        filtered_gains = (self.alpha_gains * constrained_gains + 
                         (1.0 - self.alpha_gains) * self.prev_gains)
        
        # Step 3: IIR Filtering for CCT (usually smoother than gains)
        # CCT changes are typically more gradual, so we use lower alpha
        filtered_cct = (self.alpha_cct * result.cct + 
                       (1.0 - self.alpha_cct) * self.prev_cct)
        
        # Step 4: Update state for next frame
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
        """
        Limit the maximum change in gains per frame (convergence control).
        
        This prevents sudden jumps in AWB gains that would cause visible
        color shifts in video. It ensures smooth convergence to new AWB values.
        
        Algorithm:
            1. Calculate change: delta = new_gains - prev_gains
            2. Clip delta to [-max_gain_change, +max_gain_change]
            3. Return: prev_gains + clipped_delta
        
        Example:
            prev_gains = [1.0, 1.0, 1.0]
            new_gains = [1.5, 1.0, 0.5]  # Large change
            max_gain_change = 0.1
            
            delta = [0.5, 0.0, -0.5]  # Too large
            clipped_delta = [0.1, 0.0, -0.1]  # Clipped
            result = [1.1, 1.0, 0.9]  # Smooth change
        
        Args:
            new_gains: New AWB gains from fusion
        
        Returns:
            Constrained gains (limited change from previous frame)
        """
        if self.prev_gains is None:
            return new_gains  # First frame, no constraint needed
        
        # Calculate change from previous frame
        delta = new_gains - self.prev_gains
        
        # Clip delta to maximum allowed change per frame
        # This ensures gradual convergence, not sudden jumps
        clipped_delta = np.clip(delta, -self.max_gain_change, self.max_gain_change)
        
        # Return previous gains plus constrained change
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
    
    This is the primary interface for AWB processing. It coordinates all
    components: statistics extraction, multiple analyzers, fusion, temporal
    filtering, and CCM selection.
    
    Complete Processing Pipeline:
        1. Extract frame statistics (from image or sensor data)
        2. Run all analyzers in parallel
        3. Fuse analyzer results using trust-based weighting
        4. Apply temporal filtering (for video smoothness)
        5. Select Color Correction Matrix (CCM)
        6. Apply scene-specific preferences
        7. Return final AWB output
    
    Default Configuration:
        - Analyzers: Heuristic, Face-aware, Segmentation, ML
        - Fusion: Weighted average by trust
        - Temporal: α=0.3 for gains, α=0.2 for CCT
    
    Usage:
        # Initialize system
        awb = AutoWhiteBalanceSystem()
        
        # Process single image
        result = awb.process_frame(image, bv=5.0, scene="portrait")
        
        # Apply correction
        corrected = image.copy()
        corrected[:, :, 0] *= result.gains[2]  # Blue
        corrected[:, :, 1] *= result.gains[1]  # Green
        corrected[:, :, 2] *= result.gains[0]  # Red
        
        # Check performance
        stats = awb.get_performance_stats()
        print(f"Average: {stats['average_time']:.2f}ms")
    """
    
    def __init__(self, analyzers: Optional[List[AWBAnalyzer]] = None,
                 fusion_strategy: str = "weighted_average",
                 temporal_alpha_gains: float = 0.3,
                 temporal_alpha_cct: float = 0.2):
        """
        Initialize AWB system.
        
        Args:
            analyzers: List of AWB analyzers to use. If None, uses default set:
                      [HeuristicAnalyzer, FaceAwareAnalyzer, 
                       SegmentationAnalyzer, MLAnalyzer]
            fusion_strategy: How to combine analyzer results
                           - "weighted_average": Weight by trust (default)
                           - "trust_threshold": Only high-confidence analyzers
            temporal_alpha_gains: Smoothing factor for gains (0.1-0.5)
                                 Lower = smoother video, Higher = faster response
            temporal_alpha_cct: Smoothing factor for CCT (typically 0.1-0.3)
                               Usually lower than alpha_gains
        """
        # Initialize analyzers
        # Default set provides good coverage: heuristic (general), face (portraits),
        # segmentation (complex scenes), ML (learning-based)
        if analyzers is None:
            self.analyzers = [
                HeuristicAnalyzer(),      # Gray world - general purpose
                FaceAwareAnalyzer(),      # Skin tone optimization
                SegmentationAnalyzer(),   # Region-based correction
                MLAnalyzer()              # Machine learning approach
            ]
        else:
            self.analyzers = analyzers
        
        # Initialize core components
        self.stats_extractor = FrameStatsExtractor()  # Extract RGB statistics
        self.fusion = HybridDecisionFusion(fusion_strategy)  # Combine analyzers
        self.temporal_filter = TemporalFilter(
            alpha_gains=temporal_alpha_gains,
            alpha_cct=temporal_alpha_cct
        )  # Smooth video transitions
        self.ccm_selector = CCMSelector()  # Select color correction matrix
        
        # Performance tracking (for monitoring and optimization)
        self.processing_times = []  # Track processing time per frame
        self.frame_count = 0        # Total frames processed
    
    def process_frame(self, frame_input: Union[np.ndarray, FrameSummary],
                     **kwargs) -> AWBSystemOutput:
        """
        Process a single frame through the complete AWB pipeline.
        
        This is the main entry point for AWB processing. It orchestrates
        the entire pipeline from input to final output.
        
        Pipeline Steps:
            1. Extract statistics (if image provided) or use FrameSummary
            2. Run all analyzers in parallel
            3. Fuse analyzer results using trust-based weighting
            4. Apply temporal filtering (smooth video transitions)
            5. Select Color Correction Matrix (CCM)
            6. Apply scene-specific preferences
            7. Package and return final output
        
        Args:
            frame_input: Either:
                - np.ndarray: Input image in BGR format (H x W x 3)
                - FrameSummary: Pre-extracted statistics (more efficient)
            **kwargs: Additional frame metadata:
                - bv: Brightness value from auto-exposure (-6 to +15)
                - scene: Scene type ("portrait", "landscape", "indoor", etc.)
                - face_rois: List of face regions [(x, y, w, h), ...]
                - segmentation_map: Semantic segmentation map
                - spectral_data: Spectral sensor data
                - depth_info: Depth information
        
        Returns:
            AWBSystemOutput containing:
                - gains: Final AWB gains [R, G, B] ready to apply
                - cct: Estimated color temperature
                - ccm: Color Correction Matrix (3x3)
                - trust: Overall confidence
                - analyzer_results: Individual analyzer outputs (for debugging)
                - processing_time: Time taken (milliseconds)
        
        Example:
            # Process image
            result = awb.process_frame(
                image,
                bv=5.0,
                scene="portrait",
                face_rois=[(100, 100, 80, 80)]
            )
            
            # Apply gains
            corrected = image.copy()
            corrected[:, :, 0] *= result.gains[2]  # Blue
            corrected[:, :, 1] *= result.gains[1]  # Green
            corrected[:, :, 2] *= result.gains[0]  # Red
        """
        start_time = time.time()  # Start performance timer
        
        # Step 1: Extract or prepare frame summary
        # If FrameSummary provided, use it directly (more efficient)
        # Otherwise, extract statistics from image
        if isinstance(frame_input, FrameSummary):
            frame = frame_input  # Already extracted, use directly
        else:
            # Extract statistics from full image
            # This downsamples to grid size (e.g., 64x48) for efficiency
            frame = self.stats_extractor.extract_from_image(frame_input, **kwargs)
        
        # Step 2: Run all analyzers in parallel
        # Each analyzer provides an independent AWB decision
        analyzer_results = []
        for analyzer in self.analyzers:
            try:
                result = analyzer.analyze(frame)
                analyzer_results.append(result)
            except Exception as e:
                # If analyzer fails, log warning but continue with others
                # This makes the system robust to individual analyzer failures
                print(f"Warning: Analyzer {analyzer.name} failed: {e}")
                continue
        
        # Step 3: Fuse analyzer results
        # Combine multiple decisions into one using trust-based weighting
        if analyzer_results:
            fused_result = self.fusion.fuse(analyzer_results, frame)
        else:
            # Fallback: If all analyzers fail, return neutral gains
            # This should rarely happen, but ensures system always returns valid output
            fused_result = AnalyzerResult(
                gains=np.array([1.0, 1.0, 1.0], dtype=np.float32),  # No correction
                cct=4000.0,  # Neutral CCT
                trust=0.1,   # Low trust (uncertain)
                tag="fallback"
            )
        
        # Step 4: Apply temporal filtering
        # Smooth transitions between frames (critical for video)
        # This prevents color flickering
        filtered_result = self.temporal_filter.filter(fused_result)
        
        # Step 5: Select Color Correction Matrix (CCM)
        # CCM maps from camera RGB to standard color space (e.g., sRGB)
        # Selection based on CCT and scene type
        ccm = self.ccm_selector.select_ccm(
            filtered_result.cct,  # Color temperature
            frame.bv,             # Brightness value
            frame.scene           # Scene type
        )
        
        # Step 6: Apply scene-specific preferences
        # Optional adjustments based on scene understanding
        # Example: Slightly warmer for portraits
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
        """
        Apply scene-specific or user preferences to AWB gains.
        
        This allows fine-tuning AWB based on scene understanding or user
        preferences. For example, portraits often look better slightly warmer.
        
        Current Adjustments:
            - Portrait: +2% red, -1% blue (warmer, more flattering skin)
            - Landscape: -1% red, +1% blue (cooler, more natural sky/water)
        
        Args:
            gains: AWB gains from temporal filtering
            frame: FrameSummary with scene information
        
        Returns:
            Adjusted gains with preferences applied
        
        Note: These are subtle adjustments (1-2%) to preserve natural look
              while adding scene-appropriate warmth/coolness.
        """
        adjusted_gains = gains.copy()
        
        # Scene-specific adjustments
        if frame.scene == 'portrait':
            # Slightly warmer for portraits (more flattering skin tones)
            # Boost red slightly, reduce blue slightly
            adjusted_gains *= np.array([1.02, 1.0, 0.99], dtype=np.float32)
        elif frame.scene == 'landscape':
            # Slightly cooler for landscapes (more natural sky/water)
            # Reduce red slightly, boost blue slightly
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
