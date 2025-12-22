# Chapter 7: Auto White Balance (AWB) System

## Abstract

This chapter presents a comprehensive Auto White Balance (AWB) system designed for computational photography applications. The system employs a multi-analyzer framework that combines heuristic methods, face-aware processing, segmentation-based analysis, and machine learning approaches to robustly estimate scene illumination. A trust-based fusion mechanism intelligently combines analyzer outputs, while temporal filtering ensures smooth, flicker-free video transitions. The system adapts to diverse lighting conditions through intelligent Color Correction Matrix (CCM) selection. Optimized for real-time mobile and embedded systems, the architecture is extensible, allowing easy integration of new analyzers and fusion strategies.

**Keywords**: Auto White Balance, AWB, Color Correction, Multi-Analyzer, Trust-Based Fusion, Temporal Filtering, Color Correction Matrix, Real-time Imaging

**Code Repository**: https://github.com/JeffRen1977/auto-white-balance

## 7.1 Introduction

Auto White Balance (AWB) is essential for producing natural-looking images under varying lighting conditions. Different light sources—incandescent bulbs, daylight, fluorescent lights—emit light with different color temperatures, causing images to appear too warm (yellow/orange) or too cool (blue) without correction.

This chapter introduces a production-ready AWB system that addresses these challenges through:

- **Multi-Analyzer Framework**: Combines heuristic, face-aware, segmentation-based, and ML approaches
- **Trust-Based Fusion**: Intelligently weights analyzer results by confidence
- **Temporal Filtering**: Smooths AWB decisions across video frames
- **Adaptive CCM Selection**: Chooses color correction matrices based on scene conditions
- **Real-time Optimization**: Efficient processing for mobile and embedded systems
- **Extensible Design**: Easy to add custom analyzers and fusion strategies

## 7.2 System Architecture

The AWB system follows a modular pipeline architecture:

```
Input Image/Sensor Data
    ↓
Frame Statistics Extraction (RGB grids, metadata)
    ↓
Multiple Analyzers (Parallel Processing)
    ├─ HeuristicAnalyzer
    ├─ FaceAwareAnalyzer
    ├─ SegmentationAnalyzer
    └─ MLAnalyzer
    ↓
Trust-Based Fusion (Combine results)
    ↓
Temporal Filtering (Smooth transitions)
    ↓
CCM Selection (Color correction)
    ↓
Final AWB Output (Gains, CCT, CCM)
```

### 7.2.1 Core Concepts

**AWB Gains**: Multipliers applied to RGB channels to correct color cast. For example, gains `[0.9, 1.0, 1.1]` reduce red and boost blue to correct warm lighting.

**Color Temperature (CCT)**: Measured in Kelvin, describes light color:
- Warm: < 3000K (incandescent, candle)
- Neutral: 4000-6500K (daylight, fluorescent)
- Cool: > 6500K (shade, overcast)

**Trust Scores**: Confidence values [0, 1] indicating analyzer reliability. Higher trust means more reliable results.

**Temporal Filtering**: Smooths AWB decisions using IIR filters to prevent flickering in video.

## 7.3 Implementation

### 7.3.1 Data Structures

#### FrameSummary

Container for frame statistics and metadata used by all analyzers.

```python
@dataclass
class FrameSummary:
    r_grid: np.ndarray          # Red channel statistics (subsampled grid)
    g_grid: np.ndarray          # Green channel statistics
    b_grid: np.ndarray          # Blue channel statistics
    bv: float = 0.0            # Brightness Value from Auto Exposure
    scene: Optional[str] = None # Scene type (portrait, landscape, etc.)
    face_rois: Optional[List[Tuple[int, int, int, int]]] = None
    segmentation_map: Optional[np.ndarray] = None
    spectral_data: Optional[Dict] = None
    depth_info: Optional[Dict] = None
    timestamp: float = 0.0
```

**Key Methods**:
- `global_means()`: Returns (R_mean, G_mean, B_mean) for entire frame
- `region_means(mask)`: Returns means for a specific region

**Usage**: Created by `FrameStatsExtractor` from images or sensor data.

#### AnalyzerResult

Result from a single AWB analyzer.

```python
@dataclass
class AnalyzerResult:
    gains: np.ndarray    # AWB gains [R, G, B]
    cct: float          # Estimated CCT in Kelvin
    trust: float        # Confidence score [0, 1]
    tag: str = ""       # Analyzer identifier
    metadata: Dict = None  # Additional metadata
```

**Example**: `AnalyzerResult(gains=[0.95, 1.0, 1.05], cct=3200, trust=0.8, tag="heuristic")`

#### AWBSystemOutput

Final system output containing all results.

```python
@dataclass
class AWBSystemOutput:
    gains: np.ndarray           # Final AWB gains [R, G, B]
    cct: float                 # Estimated CCT
    ccm: np.ndarray           # Color Correction Matrix (3x3)
    trust: float              # Overall confidence
    analyzer_results: List[AnalyzerResult]  # Individual results
    processing_time: float    # Processing time in ms
    metadata: Dict           # Additional metadata
```

### 7.3.2 Frame Statistics Extractor

Extracts color statistics from images or sensor data into a subsampled grid for efficient processing.

```python
class FrameStatsExtractor:
    def __init__(self, grid_width: int = 64, grid_height: int = 48):
        """Initialize with target grid size (default: 64x48)."""
    
    def extract_from_image(self, image: np.ndarray, **kwargs) -> FrameSummary:
        """
        Extract statistics from full-resolution image.
        
        Args:
            image: Input image (BGR format from OpenCV)
            **kwargs: Metadata (bv, scene, face_rois, etc.)
        
        Returns:
            FrameSummary with RGB grids and metadata
        """
    
    def extract_from_bayer_stats(self, r_stats, g_stats, b_stats, **kwargs) -> FrameSummary:
        """
        Create FrameSummary from precomputed Bayer statistics.
        Used when statistics are already available from sensor.
        """
```

**Key Features**:
- Subsampling: Processes 64×48 grid instead of full resolution (e.g., 4032×3024)
- Efficiency: Reduces data by ~4000× while preserving color statistics
- Flexibility: Works with images or precomputed sensor statistics

**Example**:
```python
extractor = FrameStatsExtractor(grid_width=64, grid_height=48)
frame = extractor.extract_from_image(image, bv=5.0, scene="portrait")
```

### 7.3.3 Analyzers

All analyzers inherit from `AWBAnalyzer` and implement the `analyze()` method.

#### AWBAnalyzer (Base Class)

```python
class AWBAnalyzer(ABC):
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def analyze(self, frame: FrameSummary) -> AnalyzerResult:
        """Analyze frame and return AWB decision."""
        pass
```

#### HeuristicAnalyzer

Implements the Gray World assumption: average scene color should be neutral gray.

```python
class HeuristicAnalyzer(AWBAnalyzer):
    def __init__(self, name: str = "heuristic"):
        super().__init__(name)
        self.gray_world_target = 0.5  # Target gray level
    
    def analyze(self, frame: FrameSummary) -> AnalyzerResult:
        """
        Process:
        1. Compute global RGB means
        2. Calculate gains to normalize to gray_world_target
        3. Estimate CCT from chromaticity
        4. Compute trust based on brightness and gain stability
        """
```

**Trust Computation**:
- Higher trust (0.6-0.9) for moderate brightness (BV 2-10)
- Lower trust (0.2-0.4) for extreme gains or unusual scenes
- Very low trust (0.1) for extreme brightness (BV < 0 or > 12)

**Use Case**: General-purpose AWB, works well for most scenes.

#### FaceAwareAnalyzer

Prioritizes accurate skin tones for portrait photography.

```python
class FaceAwareAnalyzer(AWBAnalyzer):
    def __init__(self, name: str = "face_aware"):
        super().__init__(name)
        self.skin_tone_target = 0.48  # Slightly warmer than neutral
    
    def analyze(self, frame: FrameSummary) -> AnalyzerResult:
        """
        Process:
        1. Check if face_rois are provided
        2. Extract statistics from face regions only
        3. Compute gains optimized for skin tone
        4. Estimate CCT and trust based on face quality
        """
```

**Trust Factors**:
- No faces: trust = 0.01 (essentially disabled)
- Single face: trust = 0.6-0.8 (depending on brightness)
- Multiple faces: trust = 0.8-1.0 (higher confidence)

**Use Case**: Portrait photography, video calls, face detection scenarios.

#### SegmentationAnalyzer

Applies region-specific AWB using semantic segmentation.

```python
class SegmentationAnalyzer(AWBAnalyzer):
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
        Process:
        1. Check if segmentation_map is available
        2. Analyze each segment independently
        3. Compute weighted combination based on segment importance
        4. Return fused result
        """
```

**Segment Weighting**: Different semantic regions get different weights:
- Person/Skin: High weight (0.8) - prioritize human subjects
- Sky: Medium weight (0.6) - important for landscape
- Background: Lower weight (0.3) - less critical

**Use Case**: Complex scenes with multiple regions requiring different corrections.

#### MLAnalyzer

Machine learning-based AWB (placeholder for model integration).

```python
class MLAnalyzer(AWBAnalyzer):
    def __init__(self, name: str = "ml"):
        super().__init__(name)
        self.low_light_threshold = 2.0
    
    def analyze(self, frame: FrameSummary) -> AnalyzerResult:
        """
        Current: Simulates ML behavior
        Production: Replace with actual model inference
        
        Better performance in low light (BV < 2.0)
        """
```

**Integration Points**:
1. Load TFLite/ONNX model in `__init__`
2. Run inference in `analyze()` method
3. Use frame statistics as model input
4. Return gains + CCT + trust as output

**Use Case**: Scenarios with training data, low-light conditions, specialized applications.

### 7.3.4 Hybrid Decision Fusion

Combines multiple analyzer results using trust-based weighting.

```python
class HybridDecisionFusion:
    def __init__(self, fusion_strategy: str = "weighted_average"):
        """
        Args:
            fusion_strategy: "weighted_average" or "trust_threshold"
        """
    
    def fuse(self, results: List[AnalyzerResult], 
            frame: FrameSummary) -> AnalyzerResult:
        """
        Fuse multiple analyzer results.
        
        weighted_average:
            1. Normalize trust scores to sum to 1.0
            2. Compute weighted average: Σ(trust[i] × gains[i])
            3. Compute weighted average of CCT
            4. Set trust to max(individual_trusts)
        
        trust_threshold:
            1. Filter analyzers with trust >= 0.3
            2. If none pass, use best single analyzer
            3. Otherwise, apply weighted_average to filtered set
        """
```

**Fusion Strategies**:
- `weighted_average`: Weighs all analyzers by trust (default, most common)
- `trust_threshold`: Only uses high-confidence analyzers (trust >= 0.3)

**Example**:
```python
fusion = HybridDecisionFusion(fusion_strategy="weighted_average")
results = [heuristic_result, face_result, ml_result, seg_result]
fused_result = fusion.fuse(results, frame_summary)
```

### 7.3.5 Temporal Filtering

Smooths AWB decisions across video frames to prevent flickering.

```python
class TemporalFilter:
    def __init__(self, alpha_gains: float = 0.3, 
                alpha_cct: float = 0.2,
                max_gain_change: float = 0.1):
        """
        Args:
            alpha_gains: Smoothing strength for gains (0.1-0.5)
                        Lower = smoother, Higher = faster response
            alpha_cct: Smoothing strength for CCT
            max_gain_change: Maximum gain change per frame
        """
    
    def filter(self, result: AnalyzerResult) -> AnalyzerResult:
        """
        Apply temporal filtering using IIR filter:
        
        filtered[n] = α × input[n] + (1-α) × filtered[n-1]
        
        Also applies convergence control to limit per-frame changes.
        """
```

**Parameter Guidelines**:
- **Photo mode**: α = 0.5 (faster convergence, no flicker concern)
- **Video mode**: α = 0.2-0.3 (smoother transitions)
- **Low light**: α = 0.2 (prevent flicker)
- **Stable scenes**: α = 0.15 (very smooth)

**Convergence Control**: `max_gain_change` limits per-frame gain steps to prevent sudden jumps.

**Example**:
```python
temporal_filter = TemporalFilter(
    alpha_gains=0.3,     # Moderate smoothing
    alpha_cct=0.2,       # Stronger CCT smoothing
    max_gain_change=0.1 # Limit gain step to 0.1 per frame
)

# In video processing loop
for frame in video_frames:
    result = awb_system.process_frame(frame)
    smoothed_result = temporal_filter.filter(result)
    apply_awb(frame, smoothed_result.gains)
```

### 7.3.6 Color Correction Matrix (CCM) Selection

Selects scene-adaptive color correction matrix based on CCT and lighting conditions.

```python
class CCMSelector:
    def __init__(self):
        """Initialize with predefined CCM database."""
        self.ccm_database = {
            'tungsten': np.array([...]),    # < 3000K
            'fluorescent': np.array([...]), # 3000-4000K
            'daylight': np.array([...]),   # 4000-6500K
            'cloudy': np.array([...])      # > 6500K
        }
    
    def select_ccm(self, cct: float, bv: float, 
                  scene: Optional[str] = None) -> np.ndarray:
        """
        Select appropriate CCM based on conditions.
        
        CCT to Illuminant Mapping:
        - < 3000K: Tungsten
        - 3000-4000K: Fluorescent
        - 4000-6500K: Daylight
        - > 6500K: Cloudy
        
        Returns: 3x3 CCM matrix
        """
```

**Scene Adjustments**:
- **Portrait**: Slightly warmer (↑red, ↓blue)
- **Landscape**: Slightly cooler (↓red, ↑blue)
- **Low light** (BV < 2): Reduce red, boost blue slightly

**Example**:
```python
ccm_selector = CCMSelector()
ccm = ccm_selector.select_ccm(cct=5500, bv=6.0, scene="portrait")

# Apply CCM to image
corrected_rgb = np.dot(rgb_image.reshape(-1, 3), ccm.T).reshape(rgb_image.shape)
```

### 7.3.7 Main AWB System

Complete AWB pipeline orchestrator.

```python
class AutoWhiteBalanceSystem:
    def __init__(self, 
                analyzers: Optional[List[AWBAnalyzer]] = None,
                fusion_strategy: str = "weighted_average",
                temporal_alpha_gains: float = 0.3,
                temporal_alpha_cct: float = 0.2):
        """
        Initialize AWB system.
        
        Default analyzers: [Heuristic, Face, Segmentation, ML]
        """
    
    def process_frame(self, 
                     frame_input: Union[np.ndarray, FrameSummary],
                     **kwargs) -> AWBSystemOutput:
        """
        Complete AWB pipeline:
        1. Extract/prepare frame statistics
        2. Run all analyzers in parallel
        3. Fuse analyzer results
        4. Apply temporal filtering
        5. Select CCM
        6. Apply scene preferences
        7. Package output
        """
```

**Complete Example**:
```python
from src.awb_system import AutoWhiteBalanceSystem
import cv2

# Initialize system
awb = AutoWhiteBalanceSystem(
    fusion_strategy="weighted_average",
    temporal_alpha_gains=0.3,
    temporal_alpha_cct=0.2
)

# Process image
image = cv2.imread('photo.jpg')
result = awb.process_frame(
    image,
    bv=5.0,
    scene="portrait",
    face_rois=[(100, 100, 80, 80)]
)

# Access results
print(f"AWB Gains: {result.gains}")
print(f"CCT: {result.cct:.1f}K")
print(f"Trust: {result.trust:.3f}")

# Apply to image
corrected = image.copy().astype(np.float32)
corrected[:, :, 0] *= result.gains[2]  # Blue
corrected[:, :, 1] *= result.gains[1]  # Green
corrected[:, :, 2] *= result.gains[0]  # Red
corrected = np.clip(corrected, 0, 255).astype(np.uint8)
```

## 7.4 Configuration and Usage

### 7.4.1 Analyzer Configuration

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
ml_analyzer.low_light_threshold = 2.0
```

### 7.4.2 System Configuration

```python
awb_system = AutoWhiteBalanceSystem(
    analyzers=[heuristic_analyzer, face_analyzer, seg_analyzer, ml_analyzer],
    fusion_strategy="weighted_average",  # or "trust_threshold"
    temporal_alpha_gains=0.3,           # Temporal filtering strength
    temporal_alpha_cct=0.2
)
```

### 7.4.3 Real-time Configuration

For mobile/embedded systems requiring fast processing:

```python
# Real-time configuration (fewer analyzers, faster convergence)
awb_system = AutoWhiteBalanceSystem(
    analyzers=[HeuristicAnalyzer(), FaceAwareAnalyzer()],  # Fewer analyzers
    temporal_alpha_gains=0.5,  # Faster convergence
    temporal_alpha_cct=0.4
)
```

### 7.4.4 High Quality Configuration

For scenarios prioritizing quality over speed:

```python
# High quality configuration (all analyzers, smoother transitions)
awb_system = AutoWhiteBalanceSystem(
    analyzers=[HeuristicAnalyzer(), FaceAwareAnalyzer(), 
               SegmentationAnalyzer(), MLAnalyzer()],
    temporal_alpha_gains=0.2,  # Slower, smoother transitions
    temporal_alpha_cct=0.1
)
```

## 7.5 Integration with Camera Systems

### 7.5.1 Input Data Sources

The system supports multiple input sources:

- **Bayer Statistics**: Direct from image sensor (most efficient)
- **Full Images**: Processed through OpenCV
- **Sensor Data**: Gyroscope, GPS, spectral sensors
- **Scene Analysis**: Face detection, segmentation maps

### 7.5.2 Processing RAW Sensor Data

For RAW sensor data (e.g., RAW10 format):

```python
# Extract Bayer statistics from RAW data
r_grid, g_grid, b_grid = extract_bayer_statistics(raw_image, pattern='RGGB')

# Create frame summary
frame = FrameSummary(
    r_grid=r_grid,
    g_grid=g_grid,
    b_grid=b_grid,
    bv=estimate_brightness_value(raw_image),
    scene="auto"
)

# Process with AWB
result = awb_system.process_frame(frame)

# Apply gains to RAW data before debayering
corrected_raw = apply_awb_to_raw(raw_image, result.gains, pattern='RGGB')
```

### 7.5.3 Output Integration

```python
# Get AWB output
result = awb_system.process_frame(image, **metadata)

# Apply gains to image
corrected_image = apply_awb_gains(image, result.gains)

# Use CCM for color correction
color_corrected_image = apply_ccm(corrected_image, result.ccm)
```

## 7.6 Testing and Validation

### 7.6.1 Real Data Testing

The system includes comprehensive testing with real RAW sensor data:

```python
# Test with real RAW10 data
python3 tests/test_real_data.py
```

This script:
1. Reads RAW10 sensor data from `data/` directory
2. Extracts Bayer statistics
3. Processes through AWB system
4. Applies corrections
5. Saves results to `results/awb_real_data_results/`

### 7.6.2 Performance Monitoring

```python
# Get performance statistics
stats = awb_system.get_performance_stats()
print(f"Average processing time: {stats['average_time']:.2f}ms")
print(f"Estimated FPS: {stats['fps_estimate']:.1f}")
```

### 7.6.3 Basic Usage Example

```python
# Simple example
from src.awb_system import AutoWhiteBalanceSystem
import cv2

awb = AutoWhiteBalanceSystem()
image = cv2.imread('photo.jpg')
result = awb.process_frame(image, bv=5.0, scene="auto")

# Apply correction
corrected = image.copy().astype(np.float32)
corrected[:, :, 0] *= result.gains[2]  # Blue
corrected[:, :, 1] *= result.gains[1]  # Green
corrected[:, :, 2] *= result.gains[0]  # Red
corrected = np.clip(corrected, 0, 255).astype(np.uint8)
```

## 7.7 Key Takeaways

1. **Multi-Analyzer Approach**: Combining multiple algorithms provides robustness across diverse scenarios.

2. **Trust-Based Fusion**: Weighting analyzer results by confidence improves overall accuracy.

3. **Temporal Smoothing**: Essential for video applications to prevent flickering.

4. **Efficient Processing**: Subsampled statistics (64×48 grid) enable real-time performance.

5. **Extensibility**: Easy to add custom analyzers by inheriting from `AWBAnalyzer`.

6. **Production Ready**: Handles real sensor data, includes comprehensive error handling, and provides performance monitoring.

## 7.8 Further Reading

- **Code Repository**: https://github.com/JeffRen1977/auto-white-balance
- **Documentation**: See `docs/` directory for detailed guides
- **Examples**: `examples/basic_usage.py` for simple usage
- **Real Data Testing**: `tests/test_real_data.py` for RAW sensor processing

---

**Next Steps**: Experiment with different analyzer combinations, adjust temporal filtering parameters, and integrate with your camera pipeline.

