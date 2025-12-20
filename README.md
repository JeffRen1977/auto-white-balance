# Auto White Balance System

A comprehensive implementation of Auto White Balance (AWB) for computational photography, designed for educational purposes and production use.

## 📚 For Book Readers

This codebase accompanies the computational photography book chapter on Auto White Balance. It provides:

- **Complete AWB implementation** with multiple algorithms
- **Real-world examples** using actual sensor data
- **Educational documentation** explaining concepts step-by-step
- **Production-ready code** suitable for camera systems

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or use the setup script
./examples/setup_venv.sh
```

### Run Your First Example

```bash
# Test with real RAW sensor data
python tests/test_real_data.py

# Or use the runner script
./examples/run.sh test
```

## 📖 Documentation Structure

```
docs/
├── README.md              # This file - main overview
├── INTRODUCTION.md         # What is White Balance?
├── TUTORIAL.md            # Step-by-step tutorial
├── ALGORITHMS.md          # AWB algorithm explanations
├── API_REFERENCE.md       # Complete API documentation
└── EXAMPLES.md            # Code examples and use cases
```

**Start here**: Read `docs/INTRODUCTION.md` to understand white balance concepts.

## 📁 Project Structure

```
AutoWhiteBalance/
├── src/                    # Core AWB implementation
│   └── awb_system.py      # Main AWB system
├── examples/              # Example scripts
│   ├── basic_usage.py     # Simple AWB example
│   ├── analyzer_comparison.py  # Compare different analyzers
│   └── temporal_filtering.py  # Video AWB example
├── tests/                 # Test scripts
│   └── test_real_data.py  # Real RAW data testing
├── data/                  # Test data
│   └── *.raw             # RAW sensor data files
├── docs/                  # Documentation
│   ├── INTRODUCTION.md   # White balance concepts
│   ├── TUTORIAL.md       # Step-by-step guide
│   ├── ALGORITHMS.md     # Algorithm details
│   └── API_REFERENCE.md  # API documentation
├── results/               # Output results
│   └── awb_real_data_results/  # Test outputs
└── requirements.txt       # Python dependencies
```

## 🎓 Learning Path

### For Beginners

1. **Start with Concepts**: Read `docs/INTRODUCTION.md`
   - What is white balance?
   - Why do we need it?
   - How does it work?

2. **Try Simple Example**: Run `examples/basic_usage.py`
   - See AWB in action
   - Understand the basic workflow

3. **Learn Algorithms**: Read `docs/ALGORITHMS.md`
   - Gray world assumption
   - Face-aware AWB
   - Segmentation-based AWB
   - Machine learning approaches

4. **Follow Tutorial**: Read `docs/TUTORIAL.md`
   - Step-by-step implementation
   - Build your own AWB system

### For Advanced Users

1. **Study Implementation**: Read `src/awb_system.py`
   - Multi-analyzer framework
   - Trust-based fusion
   - Temporal filtering

2. **Test with Real Data**: Run `tests/test_real_data.py`
   - Process actual RAW sensor data
   - See production-quality results

3. **Customize**: Modify analyzers and fusion strategies
   - Add your own algorithms
   - Integrate with camera systems

## 🔬 Core Concepts

### What is Auto White Balance?

White balance corrects color casts caused by different light sources. The same white object appears:
- **Yellow** under incandescent light (warm, ~2700K)
- **White** under daylight (neutral, ~5500K)
- **Blue** under shade (cool, ~7000K)

AWB automatically adjusts RGB gains to make white objects appear white.

### How It Works

```
1. Capture → 2. Analyze → 3. Correct → 4. Display
   RAW data    Color stats   Apply gains   Final image
```

**Example**:
- Input: Scene lit by warm light (yellow cast)
- Analysis: Detects too much red/yellow
- Correction: Reduces red gain, increases blue gain
- Output: Neutral colors, white appears white

## 🛠️ Key Features

### Multi-Analyzer Framework

The system uses multiple analyzers and intelligently combines their results:

- **Heuristic Analyzer**: Gray world assumption
- **Face-Aware Analyzer**: Prioritizes skin tones
- **Segmentation Analyzer**: Region-specific correction
- **ML Analyzer**: Learning-based approach

### Trust-Based Fusion

Each analyzer provides:
- **Gains**: RGB correction values
- **CCT**: Color temperature estimate
- **Trust**: Confidence score (0-1)

The system combines analyzers weighted by their trust scores.

### Temporal Filtering

For video, smooth transitions prevent color flickering:
- IIR filter for gradual changes
- Convergence control for stability
- Scene change detection

## 📊 Example Results

### Real Sensor Data Test

**Input**: RAW10 sensor data (4032×3024, 12.2 MP)
**Processing Time**: 0.19ms (5,263 FPS capable)
**AWB Gains**: R=1.044, G=1.0, B=0.971
**Color Temperature**: 4726K (slightly cool daylight)

See `results/awb_real_data_results/` for visual comparisons.

## 📝 Code Examples

### Basic Usage

```python
from src.awb_system import AutoWhiteBalanceSystem
import cv2

# Initialize
awb = AutoWhiteBalanceSystem()

# Process image
image = cv2.imread('photo.jpg')
result = awb.process_frame(image, bv=5.0, scene="portrait")

# Apply correction
corrected = image.copy()
corrected[:, :, 0] *= result.gains[2]  # Blue
corrected[:, :, 1] *= result.gains[1]    # Green
corrected[:, :, 2] *= result.gains[0]   # Red

print(f"Gains: {result.gains}")
print(f"CCT: {result.cct:.1f}K")
```

### Custom Analyzers

```python
from src.awb_system import HeuristicAnalyzer, FaceAwareAnalyzer

# Use only specific analyzers
analyzers = [
    HeuristicAnalyzer(name="gray_world"),
    FaceAwareAnalyzer(name="face_priority")
]

awb = AutoWhiteBalanceSystem(analyzers=analyzers)
result = awb.process_frame(image, face_rois=[(100, 100, 80, 80)])
```

## 🔍 Understanding the Output

### AWB Gains

Gains are multipliers for each color channel:
- `[1.0, 1.0, 1.0]` = No correction needed
- `[1.2, 1.0, 0.9]` = Boost red, reduce blue (warm scene)
- `[0.9, 1.0, 1.1]` = Reduce red, boost blue (cool scene)

### Color Temperature (CCT)

Measured in Kelvin:
- **2700-3000K**: Warm (incandescent)
- **4000-5000K**: Neutral (fluorescent)
- **5500-6500K**: Daylight
- **7000K+**: Cool (shade/overcast)

### Trust Score

Confidence in the AWB decision (0.0 to 1.0):
- **> 0.7**: High confidence
- **0.4-0.7**: Moderate confidence
- **< 0.4**: Low confidence (may need manual adjustment)

## 📚 Further Reading

- **Introduction**: `docs/INTRODUCTION.md` - White balance fundamentals
- **Tutorial**: `docs/TUTORIAL.md` - Build your own AWB
- **Algorithms**: `docs/ALGORITHMS.md` - Algorithm deep dive
- **API Reference**: `docs/API_REFERENCE.md` - Complete API docs
- **Examples**: `docs/EXAMPLES.md` - More code examples

## 🧪 Testing

### Run Tests

```bash
# Test with real RAW data
python tests/test_real_data.py

# Check results
open results/awb_real_data_results/comparison.jpg
```

### Expected Output

- Original and corrected images
- Side-by-side comparison
- Detailed JSON results with all analyzer outputs

## 🤝 Contributing

This code is provided for educational purposes. Feel free to:
- Modify and experiment
- Add your own analyzers
- Integrate with camera systems
- Share improvements

## 📄 License

See LICENSE file for details.

## 🙏 Acknowledgments

This implementation is based on advanced AWB systems used in modern computational photography, adapted for educational clarity.

---

**For book readers**: Start with `docs/INTRODUCTION.md` to begin your white balance journey!
