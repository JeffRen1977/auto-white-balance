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

## Core Components

### 1. Frame Statistics Extractor

Extracts color statistics from images or sensor data:

```python
from awb_system import FrameStatsExtractor

extractor = FrameStatsExtractor(grid_width=64, grid_height=48)
frame_summary = extractor.extract_from_image(image, bv=5.0, scene="outdoor")
```

### 2. Analyzers

#### Heuristic Analyzer
Based on gray world assumption and daylight locus:

```python
from awb_system import HeuristicAnalyzer

analyzer = HeuristicAnalyzer()
result = analyzer.analyze(frame_summary)
```

#### Face-Aware Analyzer
Prioritizes skin tone accuracy:

```python
from awb_system import FaceAwareAnalyzer

analyzer = FaceAwareAnalyzer()
result = analyzer.analyze(frame_summary)
```

#### Segmentation Analyzer
Applies different corrections to different image regions:

```python
from awb_system import SegmentationAnalyzer

analyzer = SegmentationAnalyzer()
result = analyzer.analyze(frame_summary)
```

#### ML Analyzer
Machine learning-based approach (placeholder implementation):

```python
from awb_system import MLAnalyzer

analyzer = MLAnalyzer()
result = analyzer.analyze(frame_summary)
```

### 3. Hybrid Decision Fusion

Combines multiple analyzer results:

```python
from awb_system import HybridDecisionFusion

fusion = HybridDecisionFusion(fusion_strategy="weighted_average")
fused_result = fusion.fuse(analyzer_results, frame_summary)
```

### 4. Temporal Filtering

Smooths AWB decisions across frames:

```python
from awb_system import TemporalFilter

temporal_filter = TemporalFilter(alpha_gains=0.3, alpha_cct=0.2)
smoothed_result = temporal_filter.filter(analyzer_result)
```

### 5. Color Correction Matrix Selection

Selects appropriate CCM based on scene characteristics:

```python
from awb_system import CCMSelector

ccm_selector = CCMSelector()
ccm_matrix = ccm_selector.select_ccm(cct=5500, bv=6.0, scene="portrait")
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
