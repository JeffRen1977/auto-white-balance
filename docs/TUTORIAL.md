# Step-by-Step Tutorial: Building an AWB System

This tutorial walks you through building an Auto White Balance system from scratch, explaining each component.

## Prerequisites

- Basic Python knowledge
- Understanding of RGB color space
- NumPy basics

## Tutorial Overview

We'll build an AWB system in steps:
1. Basic gray world AWB
2. Add CCT estimation
3. Add trust scoring
4. Add multiple analyzers
5. Add fusion
6. Add temporal filtering

## Step 1: Basic Gray World AWB

The simplest AWB algorithm assumes the average scene color should be neutral gray.

### Implementation

```python
import numpy as np

def compute_awb_gains(image):
    """
    Compute AWB gains using gray world assumption.
    
    Steps:
    1. Compute mean RGB values
    2. Calculate gains to normalize to gray
    3. Return gains
    """
    # Compute mean for each channel
    r_mean = np.mean(image[:, :, 0])  # Red
    g_mean = np.mean(image[:, :, 1])   # Green
    b_mean = np.mean(image[:, :, 2])  # Blue
    
    # Gray world: average should be gray
    # We want: R_mean × R_gain = G_mean × G_gain = B_mean × B_gain
    # Set G_gain = 1.0 (reference)
    g_gain = 1.0
    r_gain = g_mean / (r_mean + 1e-8)  # Avoid division by zero
    b_gain = g_mean / (b_mean + 1e-8)
    
    return np.array([r_gain, g_gain, b_gain])

# Test
image = np.array([[[200, 150, 100]]])  # Warm image (more red)
gains = compute_awb_gains(image)
print(f"Gains: {gains}")  # Should reduce red, boost blue
```

### Understanding the Math

**Gray World Principle**:
- Average scene color: `(R_mean, G_mean, B_mean)`
- After correction: `(R_mean × R_gain, G_mean × G_gain, B_mean × B_gain)`
- We want: `R_mean × R_gain = G_mean × G_gain = B_mean × B_gain`

**Solution**:
- Set `G_gain = 1.0` (reference)
- `R_gain = G_mean / R_mean` (normalize red to green)
- `B_gain = G_mean / B_mean` (normalize blue to green)

## Step 2: Apply Gains to Image

Now let's apply the gains to correct the image.

```python
def apply_awb(image, gains):
    """
    Apply AWB gains to image.
    
    Args:
        image: Input image (H, W, 3) in BGR format
        gains: AWB gains [R_gain, G_gain, B_gain]
    
    Returns:
        Corrected image
    """
    corrected = image.copy().astype(np.float32)
    
    # Apply gains to each channel
    corrected[:, :, 0] *= gains[2]  # Blue channel
    corrected[:, :, 1] *= gains[1]   # Green channel
    corrected[:, :, 2] *= gains[0]   # Red channel
    
    # Clip to valid range [0, 255]
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)
    
    return corrected

# Complete example
image = cv2.imread('warm_image.jpg')
gains = compute_awb_gains(image)
corrected = apply_awb(image, gains)
cv2.imwrite('corrected.jpg', corrected)
```

## Step 3: Estimate Color Temperature

Color temperature helps us understand the lighting conditions.

```python
def estimate_cct(r_mean, g_mean, b_mean):
    """
    Estimate Color Temperature from RGB means.
    
    Simple estimation based on blue ratio.
    More blue → higher CCT (cooler light)
    """
    # Compute chromaticity (normalized RGB)
    total = r_mean + g_mean + b_mean + 1e-8
    r_chrom = r_mean / total
    g_chrom = g_mean / total
    b_chrom = b_mean / total
    
    # Simple mapping: blue ratio → CCT
    # More blue = higher CCT (cooler)
    blue_ratio = b_chrom
    
    # Map to 2000K - 10000K range
    cct = 2000 + 8000 * blue_ratio
    
    return cct

# Example
r, g, b = 200, 150, 100  # Warm image
cct = estimate_cct(r, g, b)
print(f"CCT: {cct:.0f}K")  # Should be low (warm)
```

## Step 4: Add Trust Scoring

Not all scenes are suitable for gray world. We need confidence scores.

```python
def compute_trust(image, gains, bv=5.0):
    """
    Compute trust score for AWB decision.
    
    Factors:
    - Brightness: Well-lit scenes are more reliable
    - Gain stability: Extreme gains suggest unusual scene
    - Scene type: Some scenes are easier than others
    """
    # Brightness factor (BV: -6 to +15)
    brightness_factor = np.clip((bv + 6) / 12.0, 0.1, 0.9)
    
    # Gain stability (extreme gains = low trust)
    gain_std = np.std(gains)
    gain_mean = np.mean(gains)
    stability = 1.0 - (gain_std / (gain_mean + 1e-8))
    stability = np.clip(stability, 0.1, 1.0)
    
    # Combined trust
    trust = brightness_factor * stability
    
    return trust

# Example
gains = np.array([0.9, 1.0, 1.1])
trust = compute_trust(image, gains, bv=6.0)
print(f"Trust: {trust:.2f}")  # 0.0 to 1.0
```

## Step 5: Multiple Analyzers

Different algorithms work better in different scenarios.

```python
class Analyzer:
    """Base class for AWB analyzers."""
    
    def analyze(self, image):
        """Return (gains, cct, trust)."""
        raise NotImplementedError

class GrayWorldAnalyzer(Analyzer):
    """Gray world assumption."""
    
    def analyze(self, image):
        gains = compute_awb_gains(image)
        r, g, b = np.mean(image, axis=(0, 1))
        cct = estimate_cct(r, g, b)
        trust = compute_trust(image, gains)
        return gains, cct, trust

class FaceAwareAnalyzer(Analyzer):
    """Face-aware AWB."""
    
    def analyze(self, image, face_roi=None):
        if face_roi is None:
            return np.array([1.0, 1.0, 1.0]), 4000.0, 0.01
        
        # Extract face region
        x, y, w, h = face_roi
        face_region = image[y:y+h, x:x+w]
        
        # Compute gains from face only
        gains = compute_awb_gains(face_region)
        r, g, b = np.mean(face_region, axis=(0, 1))
        cct = estimate_cct(r, g, b)
        trust = 0.8  # High trust with face
        
        return gains, cct, trust

# Use multiple analyzers
gray_analyzer = GrayWorldAnalyzer()
face_analyzer = FaceAwareAnalyzer()

gray_gains, gray_cct, gray_trust = gray_analyzer.analyze(image)
face_gains, face_cct, face_trust = face_analyzer.analyze(image, face_roi=(100, 100, 80, 80))
```

## Step 6: Fusion

Combine multiple analyzer results intelligently.

```python
def fuse_results(results):
    """
    Fuse multiple analyzer results using trust weighting.
    
    Args:
        results: List of (gains, cct, trust) tuples
    
    Returns:
        Fused (gains, cct, trust)
    """
    if not results:
        return np.array([1.0, 1.0, 1.0]), 4000.0, 0.1
    
    # Extract components
    all_gains = np.array([r[0] for r in results])
    all_ccts = np.array([r[1] for r in results])
    all_trusts = np.array([r[2] for r in results])
    
    # Normalize trusts to sum to 1.0
    total_trust = np.sum(all_trusts)
    if total_trust == 0:
        weights = np.ones(len(results)) / len(results)
    else:
        weights = all_trusts / total_trust
    
    # Weighted average
    fused_gains = np.sum(weights[:, None] * all_gains, axis=0)
    fused_cct = np.sum(weights * all_ccts)
    fused_trust = np.max(all_trusts)  # Use highest trust
    
    return fused_gains, fused_cct, fused_trust

# Example
results = [
    (gray_gains, gray_cct, gray_trust),
    (face_gains, face_cct, face_trust)
]
fused_gains, fused_cct, fused_trust = fuse_results(results)
```

## Step 7: Temporal Filtering (for Video)

Smooth transitions between frames to prevent flickering.

```python
class TemporalFilter:
    """Smooth AWB across video frames."""
    
    def __init__(self, alpha=0.3):
        """
        Args:
            alpha: Smoothing strength (0.1-0.5)
                   Lower = smoother, Higher = faster response
        """
        self.alpha = alpha
        self.prev_gains = None
    
    def filter(self, current_gains):
        """Apply temporal filtering."""
        if self.prev_gains is None:
            self.prev_gains = current_gains
            return current_gains
        
        # IIR filter: new = α×current + (1-α)×previous
        filtered = self.alpha * current_gains + (1 - self.alpha) * self.prev_gains
        self.prev_gains = filtered
        
        return filtered

# Video processing example
filter = TemporalFilter(alpha=0.3)

for frame in video_frames:
    gains = compute_awb_gains(frame)
    smoothed_gains = filter.filter(gains)
    corrected = apply_awb(frame, smoothed_gains)
    display(corrected)
```

## Complete Example

Putting it all together:

```python
import cv2
import numpy as np

# 1. Load image
image = cv2.imread('input.jpg')

# 2. Create analyzers
gray_analyzer = GrayWorldAnalyzer()
face_analyzer = FaceAwareAnalyzer()

# 3. Analyze
gray_result = gray_analyzer.analyze(image)
face_result = face_analyzer.analyze(image, face_roi=(100, 100, 80, 80))

# 4. Fuse
fused_gains, fused_cct, fused_trust = fuse_results([gray_result, face_result])

# 5. Apply
corrected = apply_awb(image, fused_gains)

# 6. Save
cv2.imwrite('output.jpg', corrected)

print(f"Gains: {fused_gains}")
print(f"CCT: {fused_cct:.0f}K")
print(f"Trust: {fused_trust:.2f}")
```

## Next Steps

1. **Try the code**: Run `examples/basic_usage.py`
2. **Experiment**: Modify the algorithms
3. **Read more**: `docs/ALGORITHMS.md` for deeper understanding
4. **Use the full system**: `src/awb_system.py` has production-ready implementation

## Key Takeaways

1. **Gray World**: Simple but effective for many scenes
2. **Multiple Analyzers**: Different algorithms for different scenarios
3. **Trust Weighting**: Combine results intelligently
4. **Temporal Smoothing**: Essential for video
5. **Modular Design**: Easy to add new algorithms

---

**Ready for more?** Check out `docs/ALGORITHMS.md` for algorithm details!

