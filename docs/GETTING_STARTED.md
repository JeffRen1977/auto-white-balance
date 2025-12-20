# Getting Started with Auto White Balance

Welcome! This guide will help you get started with the AWB system quickly.

## Installation

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- `numpy` - Numerical computations
- `opencv-python` - Image processing
- `scikit-image` - Additional image utilities (optional)

### Step 2: Verify Installation

```python
python -c "import numpy, cv2; print('✓ Dependencies installed')"
```

## Quick Start

### Option 1: Run Basic Example

```bash
cd examples
python basic_usage.py
```

This will:
- Create a test image with color cast
- Apply AWB correction
- Save results to `results/basic_example/`

### Option 2: Test with Real RAW Data

```bash
cd tests
python test_real_data.py
```

This will:
- Process real RAW10 sensor data
- Apply AWB correction
- Generate comparison images
- Save detailed results

### Option 3: Use the Runner Script

```bash
./examples/run.sh test
```

## Understanding the Output

### AWB Gains

Gains are multipliers for each color channel:

```python
gains = [1.044, 1.0, 0.971]  # [R, G, B]
```

**Interpretation**:
- `R=1.044`: Boost red by 4.4%
- `G=1.0`: No change (reference)
- `B=0.971`: Reduce blue by 2.9%

**Why?** The scene had a cool (blue) cast, so we:
- Increase red (warm it up)
- Decrease blue (reduce cool cast)
- Result: More neutral colors

### Color Temperature (CCT)

Measured in Kelvin:

```
2700K  ← Warm (incandescent)
4000K  ← Neutral (fluorescent)
5500K  ← Daylight
7000K  ← Cool (shade)
```

**Example**: `CCT = 4726K` means slightly cool daylight.

### Trust Score

Confidence in the AWB decision (0.0 to 1.0):

- **> 0.7**: High confidence - correction is reliable
- **0.4-0.7**: Moderate confidence - correction is reasonable
- **< 0.4**: Low confidence - may need manual adjustment

## Your First AWB Correction

### Simple Python Script

```python
import cv2
import sys
import os

# Add path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.awb_system import AutoWhiteBalanceSystem

# Load image
image = cv2.imread('your_image.jpg')

# Initialize AWB
awb = AutoWhiteBalanceSystem()

# Process
result = awb.process_frame(image, bv=5.0)

# Apply correction
corrected = image.copy().astype(np.float32)
corrected[:, :, 0] *= result.gains[2]  # Blue
corrected[:, :, 1] *= result.gains[1]   # Green
corrected[:, :, 2] *= result.gains[0]   # Red
corrected = np.clip(corrected, 0, 255).astype(np.uint8)

# Save
cv2.imwrite('corrected.jpg', corrected)

print(f"Gains: {result.gains}")
print(f"CCT: {result.cct:.0f}K")
```

## Project Structure

```
AutoWhiteBalance/
├── src/                    # Core implementation
│   └── awb_system.py      # Main AWB system
├── examples/              # Example scripts
│   ├── basic_usage.py    # Simple example
│   └── run.sh            # Runner script
├── tests/                 # Test scripts
│   └── test_real_data.py # Real data testing
├── data/                  # Test data
│   └── *.raw             # RAW sensor files
├── docs/                  # Documentation
│   ├── INTRODUCTION.md    # Concepts
│   ├── TUTORIAL.md       # Step-by-step
│   └── GETTING_STARTED.md # This file
└── results/               # Output results
    └── */                # Test outputs
```

## Learning Path

### 1. Understand Concepts (15 minutes)
Read: `docs/INTRODUCTION.md`
- What is white balance?
- Why do we need it?
- How does it work?

### 2. Try Examples (30 minutes)
Run: `examples/basic_usage.py`
- See AWB in action
- Understand the workflow
- Check the results

### 3. Follow Tutorial (1 hour)
Read: `docs/TUTORIAL.md`
- Build AWB from scratch
- Understand each component
- Learn the algorithms

### 4. Study Implementation (2 hours)
Read: `src/awb_system.py`
- See production code
- Understand architecture
- Learn best practices

### 5. Experiment (ongoing)
- Try with your images
- Modify algorithms
- Add features

## Common Tasks

### Process a Single Image

```python
from src.awb_system import AutoWhiteBalanceSystem
import cv2

awb = AutoWhiteBalanceSystem()
image = cv2.imread('photo.jpg')
result = awb.process_frame(image, bv=5.0)

# Apply gains
corrected = image.copy()
corrected[:, :, 0] *= result.gains[2]
corrected[:, :, 1] *= result.gains[1]
corrected[:, :, 2] *= result.gains[0]

cv2.imwrite('corrected.jpg', corrected)
```

### Process Video

```python
cap = cv2.VideoCapture('video.mp4')
awb = AutoWhiteBalanceSystem()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    result = awb.process_frame(frame, bv=5.0)
    
    # Apply correction
    frame[:, :, 0] *= result.gains[2]
    frame[:, :, 1] *= result.gains[1]
    frame[:, :, 2] *= result.gains[0]
    
    cv2.imshow('Corrected', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

### Use Face-Aware AWB

```python
# Detect faces first (using your face detector)
face_rois = [(100, 100, 80, 80)]  # (x, y, w, h)

result = awb.process_frame(
    image,
    bv=5.0,
    scene="portrait",
    face_rois=face_rois
)
```

## Troubleshooting

### Import Error

**Problem**: `ModuleNotFoundError: No module named 'src'`

**Solution**: Make sure you're running from the project root:
```bash
cd /path/to/AutoWhiteBalance
python examples/basic_usage.py
```

### Image Not Found

**Problem**: `FileNotFoundError`

**Solution**: Check file path:
```python
import os
print(os.path.exists('your_image.jpg'))  # Should be True
```

### Poor Results

**Problem**: AWB doesn't work well

**Solutions**:
1. Check brightness value (BV) - should be reasonable (2-10)
2. Try different analyzers
3. Use face detection for portraits
4. Check trust score - low trust means uncertain

## Next Steps

1. **Read Documentation**:
   - `docs/INTRODUCTION.md` - Concepts
   - `docs/TUTORIAL.md` - Step-by-step guide
   - `docs/API_REFERENCE.md` - Complete API

2. **Try Examples**:
   - `examples/basic_usage.py` - Simple example
   - `tests/test_real_data.py` - Real data test

3. **Experiment**:
   - Process your own images
   - Modify algorithms
   - Add features

## Getting Help

- **Documentation**: Check `docs/` folder
- **Examples**: See `examples/` folder
- **Code**: Read `src/awb_system.py` with comments

---

**Ready to dive deeper?** Start with `docs/INTRODUCTION.md`!

