# Advanced Auto White Balance (AWB) System

A comprehensive Auto White Balance implementation featuring multi-analyzer framework, trust-based fusion, temporal smoothing, and advanced color correction capabilities.

## Features

- **Multi-Analyzer Framework**: Heuristic, Face-aware, Segmentation-based, and ML analyzers
- **Trust-Based Fusion**: Intelligent combination of analyzer results
- **Temporal Filtering**: Smooth transitions and convergence control
- **Color Correction Matrix (CCM) Selection**: Scene-adaptive color correction
- **Real-time Processing**: Optimized for mobile and embedded systems
- **Extensible Architecture**: Easy to add new analyzers and fusion strategies

## System Architecture

```
Input Data → Frame Analysis → Multi-Analyzer Processing → Hybrid Fusion → Temporal Smoothing → Output
     ↓              ↓                    ↓                     ↓              ↓           ↓
Bayer Stats    Scene Detection    Heuristic/ML/Face/Seg   Weighted Avg   IIR Filter   AWB Gains
Exposure Info   Face ROIs         Analyzers               Trust-based    Convergence   CCM Matrix
Scene Labels    Segmentation      Trust Scores            Fusion         Control       Color Space
Sensor Data     Depth Info        CCT Estimation          Decision       Preferences   Transform
```

## Installation

```bash
pip install numpy opencv-python scikit-image
```

## Quick Start

### Basic Usage

```python
from awb_system import AutoWhiteBalanceSystem
import cv2

# Initialize AWB system
awb_system = AutoWhiteBalanceSystem()

# Load and process image
image = cv2.imread('test_image.jpg')
result = awb_system.process_frame(
    image,
    bv=5.0,  # Brightness value from auto-exposure
    scene="portrait",  # Scene type
    face_rois=[(100, 100, 80, 80)]  # Face regions (x, y, w, h)
)

print(f"AWB Gains: {result.gains}")
print(f"Color Temperature: {result.cct:.1f}K")
print(f"Confidence: {result.trust:.3f}")
```

### Advanced Usage with Custom Analyzers

```python
from awb_system import (
    AutoWhiteBalanceSystem, 
    HeuristicAnalyzer, 
    FaceAwareAnalyzer,
    MLAnalyzer,
    FrameStatsExtractor
)

# Create custom analyzer configuration
analyzers = [
    HeuristicAnalyzer(name="gray_world"),
    FaceAwareAnalyzer(name="face_priority"),
    MLAnalyzer(name="ml_enhanced")
]

# Initialize with custom analyzers
awb_system = AutoWhiteBalanceSystem(
    analyzers=analyzers,
    fusion_strategy="weighted_average",
    temporal_alpha_gains=0.3,
    temporal_alpha_cct=0.2
)

# Process frame with detailed metadata
result = awb_system.process_frame(
    image,
    bv=4.5,
    scene="indoor",
    face_rois=[(50, 50, 100, 100), (200, 150, 80, 80)],
    spectral_data={"illuminant_type": "fluorescent"},
    depth_info={"focus_distance": 2.5}
)

# Access individual analyzer results
for analyzer_result in result.analyzer_results:
    print(f"{analyzer_result.tag}: gains={analyzer_result.gains}, trust={analyzer_result.trust}")
```

## Core Functions and Components

### Utility Functions

Key utility functions used throughout the system:

```python
def safe_divide(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Safe division with epsilon to avoid division by zero."""
    return a / (b + eps)

def rgb_to_chromaticity(r, g, b) -> Tuple[float, float, float]:
    """Convert RGB to chromaticity coordinates (normalized)."""
    s = r + g + b + 1e-12
    return r / s, g / s, b / s

def estimate_cct_from_chromaticity(r: float, g: float, b: float) -> float:
    """
    Estimate Correlated Color Temperature (CCT) from chromaticity.
    
    Returns CCT in Kelvin (2000K - 10000K range).
    For production, use Planckian locus mapping or trained models.
    """
    
def compute_awb_gains(r_mean: float, g_mean: float, b_mean: float,
                     target_g: float = 0.5) -> np.ndarray:
    """
    Compute AWB gains to normalize color channels.
    
    Gray world assumption: average scene should be neutral gray.
    
    Args:
        r_mean, g_mean, b_mean: Mean channel values
        target_g: Target green ratio (default 0.5)
    
    Returns:
        Array of [R_gain, G_gain, B_gain] where G_gain = 1.0
    
    Example:
        Input RGB: (120, 100, 80)
        Gains: (0.833, 1.0, 1.25)
        Output: (100, 100, 100) - normalized
    """
```

## Core Components

### 1. Data Structures

#### FrameSummary
Container for frame statistics and metadata.

```python
@dataclass
class FrameSummary:
    r_grid: np.ndarray          # Red channel statistics (subsampled grid)
    g_grid: np.ndarray          # Green channel statistics
    b_grid: np.ndarray          # Blue channel statistics
    bv: float = 0.0            # Brightness Value from Auto Exposure
    scene: Optional[str] = None # Scene type from scene detection
    face_rois: Optional[List[Tuple[int, int, int, int]]] = None
    segmentation_map: Optional[np.ndarray] = None
    spectral_data: Optional[Dict] = None
    depth_info: Optional[Dict] = None
    timestamp: float = 0.0
    
    def global_means(self) -> Tuple[float, float, float]:
        """Compute global mean values for each channel."""
        
    def region_means(self, mask: np.ndarray) -> Tuple[float, float, float]:
        """Compute mean values for a specific region."""
```

#### AnalyzerResult
Result from an AWB analyzer.

```python
@dataclass
class AnalyzerResult:
    gains: np.ndarray    # AWB gains [R, G, B]
    cct: float          # Estimated CCT in Kelvin
    trust: float        # Confidence score [0, 1]
    tag: str = ""       # Analyzer identifier
    metadata: Dict = None  # Additional metadata
```

#### AWBSystemOutput
Final AWB system output with all results and metadata.

```python
@dataclass
class AWBSystemOutput:
    gains: np.ndarray           # Final AWB gains [R, G, B]
    cct: float                 # Estimated CCT
    ccm: np.ndarray           # Color Correction Matrix (3x3)
    trust: float              # Overall confidence
    analyzer_results: List[AnalyzerResult]  # Individual analyzer results
    processing_time: float    # Processing time in ms
    metadata: Dict           # Additional metadata
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
```

### 2. Frame Statistics Extractor

Extracts color statistics from images or sensor data.

```python
class FrameStatsExtractor:
    def __init__(self, grid_width: int = 64, grid_height: int = 48):
        """Initialize with target grid size for statistics."""
        
    def extract_from_image(self, image: np.ndarray, **kwargs) -> FrameSummary:
        """
        Extract statistics from a full-resolution image.
        
        Args:
            image: Input image (BGR format)
            **kwargs: Additional metadata (bv, scene, face_rois, etc.)
        
        Returns:
            FrameSummary with extracted statistics
        """
        
    def extract_from_bayer_stats(self, r_stats: np.ndarray, 
                                g_stats: np.ndarray,
                                b_stats: np.ndarray, 
                                **kwargs) -> FrameSummary:
        """
        Extract statistics from precomputed Bayer statistics.
        
        Args:
            r_stats, g_stats, b_stats: Precomputed channel statistics
            **kwargs: Additional metadata
        
        Returns:
            FrameSummary with provided statistics
        """

# Example usage
extractor = FrameStatsExtractor(grid_width=64, grid_height=48)
frame_summary = extractor.extract_from_image(image, bv=5.0, scene="outdoor")
```

### 3. Analyzers

All analyzers inherit from `AWBAnalyzer` base class and implement the `analyze()` method.

#### AWBAnalyzer (Base Class)

```python
class AWBAnalyzer(ABC):
    """Base class for AWB analyzers."""
    
    def __init__(self, name: str):
        """Initialize analyzer with a name identifier."""
        self.name = name
    
    @abstractmethod
    def analyze(self, frame: FrameSummary) -> AnalyzerResult:
        """
        Analyze frame and return AWB decision.
        
        Args:
            frame: FrameSummary containing image statistics
            
        Returns:
            AnalyzerResult with gains, CCT, trust, and metadata
        """
        pass
```

#### HeuristicAnalyzer
Gray world assumption and daylight locus principles.

```python
class HeuristicAnalyzer(AWBAnalyzer):
    """
    Heuristic-based AWB using gray world assumption.
    Assumes average scene color should be neutral gray.
    """
    
    def __init__(self, name: str = "heuristic"):
        super().__init__(name)
        self.gray_world_target = 0.5  # Target gray level
    
    def analyze(self, frame: FrameSummary) -> AnalyzerResult:
        """
        Analyze using gray world heuristic.
        
        Process:
        1. Compute global RGB means
        2. Calculate gains to normalize to gray_world_target
        3. Estimate CCT from chromaticity
        4. Compute trust based on brightness and gain stability
        
        Returns:
            AnalyzerResult with computed gains and metadata
        """

# Example usage
analyzer = HeuristicAnalyzer(name="gray_world")
result = analyzer.analyze(frame_summary)
print(f"Gains: {result.gains}, CCT: {result.cct}K, Trust: {result.trust}")
```

**Trust Computation**: Based on brightness (BV), gain stability, and scene type.
- Higher trust for moderate brightness (BV 2-10)
- Lower trust for extreme gains (scene likely unusual)

#### FaceAwareAnalyzer
Prioritizes skin tone accuracy for portrait photography.

```python
class FaceAwareAnalyzer(AWBAnalyzer):
    """
    Face-aware AWB that prioritizes skin tone accuracy.
    Uses face ROIs to compute AWB focused on skin regions.
    """
    
    def __init__(self, name: str = "face_aware"):
        super().__init__(name)
        self.skin_tone_target = 0.48  # Slightly warmer than neutral
    
    def analyze(self, frame: FrameSummary) -> AnalyzerResult:
        """
        Analyze focusing on face regions.
        
        Process:
        1. Check if face_rois are provided
        2. Extract statistics from face regions only
        3. Compute gains optimized for skin tone
        4. Estimate CCT and trust based on face quality
        
        Returns:
            AnalyzerResult with high trust if faces detected, 
            low trust (0.01) if no faces
        """

# Example usage
analyzer = FaceAwareAnalyzer()
result = analyzer.analyze(frame_summary)
# Trust will be 0.6-1.0 with faces, 0.01 without faces
```

**Trust Factors**:
- Face detection: 0.01 without faces, 0.6-1.0 with faces
- Face brightness: Higher trust for well-lit faces
- Multiple faces: Increases confidence

#### SegmentationAnalyzer
Region-specific AWB using semantic segmentation.

```python
class SegmentationAnalyzer(AWBAnalyzer):
    """
    Segmentation-based AWB for complex scenes.
    Applies different corrections to different semantic regions.
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
        """
        Analyze using segmentation information.
        
        Process:
        1. Check if segmentation_map is available
        2. Analyze each segment independently
        3. Compute weighted combination based on segment importance
        4. Return fused result
        
        Returns:
            AnalyzerResult with weighted gains from all segments,
            low trust (0.05) if no segmentation available
        """

# Example usage
analyzer = SegmentationAnalyzer()
analyzer.segment_weights[1] = 0.9  # Increase person weight
result = analyzer.analyze(frame_summary)
```

**Segment Weighting**: Different semantic regions get different weights.
- Person/Skin: High weight (0.8)
- Sky: Medium weight (0.6)
- Background: Lower weight (0.3)

#### MLAnalyzer
Machine learning-based AWB (placeholder for integration).

```python
class MLAnalyzer(AWBAnalyzer):
    """
    ML-based AWB analyzer.
    Placeholder implementation - integrate trained models here.
    """
    
    def __init__(self, name: str = "ml"):
        super().__init__(name)
        self.low_light_threshold = 2.0
    
    def analyze(self, frame: FrameSummary) -> AnalyzerResult:
        """
        Analyze using ML-based approach.
        
        Current implementation: Simulates ML behavior
        Production: Replace with actual model inference
        
        Process:
        1. Check lighting conditions
        2. Apply ML model (or simulation)
        3. Return gains with confidence
        
        Better performance in low light (BV < 2.0)
        """

# Example usage - Replace with your trained model
analyzer = MLAnalyzer(name="ml_enhanced")
# analyzer.model = load_your_model()  # Load trained model
result = analyzer.analyze(frame_summary)
```

**Integration Points**:
- Load TFLite/ONNX model in `__init__`
- Run inference in `analyze()` method
- Use frame statistics as model input
- Return gains + CCT + trust as output

### 4. Hybrid Decision Fusion

Combines multiple analyzer results using trust-based weighting.

```python
class HybridDecisionFusion:
    """
    Fuses multiple analyzer results using trust-based weighting
    and advanced fusion strategies.
    """
    
    def __init__(self, fusion_strategy: str = "weighted_average"):
        """
        Initialize fusion module.
        
        Args:
            fusion_strategy: "weighted_average" or "trust_threshold"
        """
        self.fusion_strategy = fusion_strategy
    
    def fuse(self, results: List[AnalyzerResult], 
            frame: FrameSummary) -> AnalyzerResult:
        """
        Fuse multiple analyzer results into single decision.
        
        Process (weighted_average):
        1. Normalize trust scores to sum to 1.0
        2. Compute weighted average of gains: Σ(trust[i] × gains[i])
        3. Compute weighted average of CCT
        4. Set trust to maximum analyzer trust
        
        Process (trust_threshold):
        1. Filter analyzers with trust >= threshold (0.3)
        2. If none pass, use best single analyzer
        3. Otherwise, apply weighted_average to filtered set
        
        Returns:
            Fused AnalyzerResult with combined gains and metadata
        """

# Example usage
fusion = HybridDecisionFusion(fusion_strategy="weighted_average")
results = [heuristic_result, face_result, ml_result, seg_result]
fused_result = fusion.fuse(results, frame_summary)
print(f"Fused gains: {fused_result.gains}, trust: {fused_result.trust}")
```

**Fusion Strategies**:
- `weighted_average`: Weighs all analyzers by trust (default)
- `trust_threshold`: Only uses high-confidence analyzers (trust >= 0.3)

**Trust Propagation**: Final trust = max(individual_trusts) to indicate highest confidence source.

### 5. Temporal Filtering and Convergence Control

Smooths AWB decisions across video frames to prevent flickering.

```python
class TemporalFilter:
    """
    Temporal filtering for smooth AWB transitions.
    Uses IIR filter with configurable smoothing strength.
    """
    
    def __init__(self, alpha_gains: float = 0.3, 
                alpha_cct: float = 0.2,
                max_gain_change: float = 0.1):
        """
        Initialize temporal filter.
        
        Args:
            alpha_gains: Smoothing strength for gains (0.1-0.5)
                        Lower = smoother, Higher = faster response
            alpha_cct: Smoothing strength for CCT
            max_gain_change: Maximum gain change per frame (convergence control)
        """
        self.alpha_gains = alpha_gains
        self.alpha_cct = alpha_cct
        self.max_gain_change = max_gain_change
        self.prev_gains = None
        self.prev_cct = None
        self.frame_count = 0
    
    def filter(self, result: AnalyzerResult) -> AnalyzerResult:
        """
        Apply temporal filtering to AWB result.
        
        Process:
        1. Constrain gain changes (convergence control)
        2. Apply IIR filter: new = α×current + (1-α)×previous
        3. Update state for next frame
        
        IIR Formula:
            filtered[n] = α × input[n] + (1-α) × filtered[n-1]
        
        Returns:
            Temporally filtered AnalyzerResult
        """
    
    def reset(self):
        """Reset filter state (call when scene changes)."""

# Example usage
temporal_filter = TemporalFilter(
    alpha_gains=0.3,     # Moderate smoothing
    alpha_cct=0.2,       # Stronger CCT smoothing
    max_gain_change=0.1  # Limit gain step to 0.1 per frame
)

# In video processing loop
for frame in video_frames:
    result = awb_system.process_frame(frame)
    smoothed_result = temporal_filter.filter(result)
    apply_awb(frame, smoothed_result.gains)
```

**Parameter Guidelines**:
- **Photo mode**: α=0.5 (faster convergence)
- **Video mode**: α=0.2-0.3 (smoother transitions)
- **Low light**: α=0.2 (prevent flicker)
- **Stable scenes**: α=0.15 (very smooth)

**Convergence Control**: `max_gain_change` limits per-frame gain steps to prevent sudden jumps.

### 6. Color Correction Matrix (CCM) Selection

Selects scene-adaptive color correction matrix based on CCT and lighting conditions.

```python
class CCMSelector:
    """
    Color Correction Matrix selection based on CCT and scene.
    Maps from camera RGB to standard color space (e.g., sRGB).
    """
    
    def __init__(self):
        """
        Initialize with predefined CCM database.
        
        Built-in CCMs for:
        - Tungsten (< 3000K): Warm incandescent
        - Fluorescent (3000-4000K): Office lighting
        - Daylight (4000-6500K): Natural light
        - Cloudy (> 6500K): Overcast daylight
        """
        self.ccm_database = {
            'tungsten': np.array([[1.2, -0.1, 0.0],
                                 [0.0, 1.0, 0.0],
                                 [0.0, -0.1, 1.3]]),
            'fluorescent': np.array([[1.1, 0.0, 0.0],
                                    [0.0, 1.0, 0.0],
                                    [0.0, 0.0, 1.2]]),
            'daylight': np.array([[1.0, 0.0, 0.0],
                                 [0.0, 1.0, 0.0],
                                 [0.0, 0.0, 1.0]]),
            'cloudy': np.array([[0.9, 0.0, 0.0],
                               [0.0, 1.0, 0.0],
                               [0.0, 0.0, 1.1]])
        }
    
    def select_ccm(self, cct: float, bv: float, 
                  scene: Optional[str] = None) -> np.ndarray:
        """
        Select appropriate CCM based on conditions.
        
        Process:
        1. Determine illuminant from CCT
        2. Get base CCM from database
        3. Apply scene-specific adjustments
        4. Apply brightness-based adjustments
        
        CCT to Illuminant Mapping:
        - < 3000K: Tungsten
        - 3000-4000K: Fluorescent
        - 4000-6500K: Daylight
        - > 6500K: Cloudy
        
        Args:
            cct: Color temperature in Kelvin
            bv: Brightness value
            scene: Scene type (portrait, landscape, etc.)
        
        Returns:
            3x3 CCM matrix
        """

# Example usage
ccm_selector = CCMSelector()

# Get CCM for daylight scene
ccm = ccm_selector.select_ccm(cct=5500, bv=6.0, scene="portrait")

# Apply CCM to image
corrected_rgb = np.dot(rgb_image.reshape(-1, 3), ccm.T).reshape(rgb_image.shape)
```

**Scene Adjustments**:
- **Portrait**: Slightly warmer (↑red, ↓blue)
- **Landscape**: Slightly cooler (↓red, ↑blue)
- **Low light** (BV < 2): Reduce red, boost blue slightly

### 7. Main AWB System

Complete AWB pipeline orchestrator.

```python
class AutoWhiteBalanceSystem:
    """
    Main Auto White Balance system orchestrator.
    Coordinates all components for end-to-end AWB processing.
    """
    
    def __init__(self, 
                analyzers: Optional[List[AWBAnalyzer]] = None,
                fusion_strategy: str = "weighted_average",
                temporal_alpha_gains: float = 0.3,
                temporal_alpha_cct: float = 0.2):
        """
        Initialize AWB system.
        
        Args:
            analyzers: List of AWB analyzers to use
                      If None, uses default [Heuristic, Face, Seg, ML]
            fusion_strategy: Fusion strategy for combining analyzers
            temporal_alpha_gains: Temporal smoothing for gains
            temporal_alpha_cct: Temporal smoothing for CCT
        """
        if analyzers is None:
            analyzers = [
                HeuristicAnalyzer(),
                FaceAwareAnalyzer(),
                SegmentationAnalyzer(),
                MLAnalyzer()
            ]
        
        self.analyzers = analyzers
        self.stats_extractor = FrameStatsExtractor()
        self.fusion = HybridDecisionFusion(fusion_strategy)
        self.temporal_filter = TemporalFilter(
            alpha_gains=temporal_alpha_gains,
            alpha_cct=temporal_alpha_cct
        )
        self.ccm_selector = CCMSelector()
    
    def process_frame(self, 
                     frame_input: Union[np.ndarray, FrameSummary],
                     **kwargs) -> AWBSystemOutput:
        """
        Process single frame through complete AWB pipeline.
        
        Pipeline:
        1. Extract/prepare frame statistics
        2. Run all analyzers in parallel
        3. Fuse analyzer results
        4. Apply temporal filtering
        5. Select CCM
        6. Apply scene preferences
        7. Package output
        
        Args:
            frame_input: Input image (BGR) or FrameSummary
            **kwargs: Frame metadata (bv, scene, face_rois, etc.)
        
        Returns:
            AWBSystemOutput with complete results
        """
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics (avg time, FPS, etc.)."""
    
    def reset_temporal_state(self):
        """Reset temporal filter (call on scene change)."""

# Example usage
awb_system = AutoWhiteBalanceSystem(
    fusion_strategy="weighted_average",
    temporal_alpha_gains=0.3,
    temporal_alpha_cct=0.2
)

# Process frame
result = awb_system.process_frame(
    image,
    bv=5.0,
    scene="portrait",
    face_rois=[(100, 100, 80, 80)]
)

# Access results
print(f"AWB Gains: {result.gains}")
print(f"CCT: {result.cct:.1f}K")
print(f"Trust: {result.trust:.3f}")
print(f"Processing time: {result.processing_time:.2f}ms")

# Apply to image
corrected_image = image.copy()
corrected_image[:, :, 0] *= result.gains[2]  # Blue
corrected_image[:, :, 1] *= result.gains[1]  # Green
corrected_image[:, :, 2] *= result.gains[0]  # Red
```

## Configuration Options

### Analyzer Configuration

```python
# Heuristic Analyzer
heuristic_analyzer = HeuristicAnalyzer(name="custom_heuristic")

# Face-Aware Analyzer
face_analyzer = FaceAwareAnalyzer(name="face_priority")
face_analyzer.skin_tone_target = 0.48  # Adjust skin tone preference

# Segmentation Analyzer
seg_analyzer = SegmentationAnalyzer(name="region_based")
seg_analyzer.segment_weights = {
    0: 0.3,  # Background
    1: 0.8,  # Person/Skin
    2: 0.6,  # Sky
    3: 0.5,  # Other objects
}

# ML Analyzer
ml_analyzer = MLAnalyzer(name="ml_enhanced")
ml_analyzer.low_light_threshold = 2.0  # Low light threshold
```

### System Configuration

```python
awb_system = AutoWhiteBalanceSystem(
    analyzers=[heuristic_analyzer, face_analyzer, seg_analyzer, ml_analyzer],
    fusion_strategy="weighted_average",  # or "trust_threshold"
    temporal_alpha_gains=0.3,           # Temporal filtering strength for gains
    temporal_alpha_cct=0.2              # Temporal filtering strength for CCT
)
```

## Performance Optimization

### For Real-time Processing

1. **Use subsampled statistics**: Process 64x48 grids instead of full resolution
2. **Optimize analyzer selection**: Use only necessary analyzers for your use case
3. **Adjust temporal filtering**: Lower alpha values for faster convergence

```python
# Real-time configuration
awb_system = AutoWhiteBalanceSystem(
    analyzers=[HeuristicAnalyzer(), FaceAwareAnalyzer()],  # Fewer analyzers
    temporal_alpha_gains=0.5,  # Faster convergence
    temporal_alpha_cct=0.4
)
```

### For High Quality

1. **Use all analyzers**: Enable all available analyzers
2. **Slower temporal filtering**: Lower alpha values for smoother transitions
3. **Higher resolution grids**: Use larger grid sizes

```python
# High quality configuration
awb_system = AutoWhiteBalanceSystem(
    analyzers=[HeuristicAnalyzer(), FaceAwareAnalyzer(), 
               SegmentationAnalyzer(), MLAnalyzer()],
    temporal_alpha_gains=0.2,  # Slower, smoother transitions
    temporal_alpha_cct=0.1
)
```

## Integration with Camera Systems

### Input Data Sources

The system can work with various input sources:

1. **Bayer Statistics**: Direct from image sensor
2. **Full Images**: Processed through OpenCV
3. **Sensor Data**: Gyroscope, GPS, spectral sensors
4. **Scene Analysis**: Face detection, segmentation maps

### Output Integration

```python
# Get AWB output
result = awb_system.process_frame(image, **metadata)

# Apply gains to image
corrected_image = apply_awb_gains(image, result.gains)

# Use CCM for color correction
color_corrected_image = apply_ccm(corrected_image, result.ccm)
```

## Testing and Validation

### Synthetic Testing

```python
from awb_system import create_test_image

# Create test images with known color characteristics
warm_image = create_test_image(color_temperature="warm")
cool_image = create_test_image(color_temperature="cool")

# Test AWB correction
warm_result = awb_system.process_frame(warm_image, bv=5.0)
cool_result = awb_system.process_frame(cool_image, bv=5.0)
```

### Performance Monitoring

```python
# Get performance statistics
stats = awb_system.get_performance_stats()
print(f"Average processing time: {stats['average_time']:.2f}ms")
print(f"Estimated FPS: {stats['fps_estimate']:.1f}")
```

## Extending the System

### Adding Custom Analyzers

```python
from awb_system import AWBAnalyzer, AnalyzerResult

class CustomAnalyzer(AWBAnalyzer):
    def __init__(self, name: str = "custom"):
        super().__init__(name)
        # Initialize your custom parameters
    
    def analyze(self, frame: FrameSummary) -> AnalyzerResult:
        # Implement your custom analysis logic
        gains = np.array([1.0, 1.0, 1.0])  # Your gain calculation
        cct = 5500.0  # Your CCT estimation
        trust = 0.8   # Your confidence score
        
        return AnalyzerResult(
            gains=gains,
            cct=cct,
            trust=trust,
            tag=self.name,
            metadata={'custom_info': 'value'}
        )

# Use custom analyzer
custom_analyzer = CustomAnalyzer()
awb_system = AutoWhiteBalanceSystem(analyzers=[custom_analyzer])
```

### Custom Fusion Strategies

```python
from awb_system import HybridDecisionFusion

class CustomFusion(HybridDecisionFusion):
    def fuse(self, results, frame):
        # Implement your custom fusion logic
        # For example, prioritize certain analyzers based on scene
        if frame.scene == "portrait":
            # Prioritize face analyzer
            face_results = [r for r in results if "face" in r.tag]
            if face_results:
                return face_results[0]
        
        # Fallback to default fusion
        return super().fuse(results, frame)
```

## Troubleshooting

### Common Issues

1. **Low trust scores**: Check input data quality and analyzer configuration
2. **Slow processing**: Reduce number of analyzers or increase temporal filtering alpha
3. **Color flickering**: Decrease temporal filtering alpha values
4. **Incorrect color temperature**: Verify CCT estimation and CCM selection

### Debug Information

```python
# Enable detailed output
result = awb_system.process_frame(image, **metadata)
print(f"Analyzer results: {len(result.analyzer_results)}")
for analyzer_result in result.analyzer_results:
    print(f"  {analyzer_result.tag}: trust={analyzer_result.trust:.3f}, cct={analyzer_result.cct:.1f}K")
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## Acknowledgments

This implementation is inspired by advanced AWB systems and incorporates best practices from computational photography research.
