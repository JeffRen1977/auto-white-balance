# Introduction to Auto White Balance

## What is White Balance?

White balance is a fundamental concept in digital photography and computational imaging. It ensures that colors in an image appear natural and accurate, regardless of the lighting conditions under which the photo was taken.

### The Problem

The same white object appears different colors under different light sources:

- **Incandescent bulb** (warm light): White appears **yellow/orange**
- **Daylight** (neutral light): White appears **white**
- **Shade/Overcast** (cool light): White appears **blue**

This happens because different light sources emit light with different **color temperatures**.

### Color Temperature

Color temperature is measured in **Kelvin (K)** and describes the color of light:

```
Warm Light ←───────────────→ Cool Light
2700K      4000K    5500K    7000K
(Orange)  (Neutral) (Daylight) (Blue)
```

**Common Light Sources**:
- **Candle**: ~1900K (very warm, orange)
- **Incandescent bulb**: ~2700K (warm, yellow)
- **Fluorescent**: ~4000K (neutral, slightly cool)
- **Daylight**: ~5500K (neutral, white)
- **Shade/Overcast**: ~7000K (cool, blue)
- **Blue sky**: ~10000K (very cool, blue)

### Why We Need White Balance

Without white balance correction, photos taken under different lighting conditions will have incorrect colors:

```
Scene: Person in warm indoor lighting
Without AWB: Skin looks too yellow/orange
With AWB:    Skin looks natural
```

## How Auto White Balance Works

Auto White Balance (AWB) automatically detects the lighting conditions and adjusts the image colors to make white objects appear white.

### The Process

```
1. Capture Image
   ↓
2. Analyze Color Statistics
   ↓
3. Estimate Light Source (Color Temperature)
   ↓
4. Calculate Correction Gains
   ↓
5. Apply Gains to Image
   ↓
6. Display Corrected Image
```

### Step-by-Step Example

**Scenario**: Photo taken under warm incandescent light

1. **Capture**: Camera captures image with yellow/orange cast
2. **Analyze**: System detects too much red/yellow in the image
3. **Estimate**: Determines light is warm (~3000K)
4. **Calculate**: Computes gains: reduce red, increase blue
   - Example gains: `[0.9, 1.0, 1.1]` (R, G, B)
5. **Apply**: Multiplies each pixel's RGB values by gains
6. **Result**: White objects appear white, colors are natural

## Understanding RGB Gains

AWB works by adjusting the **RGB (Red, Green, Blue)** channels independently.

### What are Gains?

Gains are **multipliers** applied to each color channel:

```python
# Original pixel: [R=200, G=150, B=100]
# Gains: [1.0, 1.0, 1.2]  (no change to R/G, boost B)

# Corrected pixel:
R_new = 200 × 1.0 = 200
G_new = 150 × 1.0 = 150
B_new = 100 × 1.2 = 120

# Result: [200, 150, 120] - more balanced
```

### Common Gain Patterns

**Warm Scene** (too much red/yellow):
- Gains: `[0.9, 1.0, 1.1]` - Reduce red, boost blue

**Cool Scene** (too much blue):
- Gains: `[1.1, 1.0, 0.9]` - Boost red, reduce blue

**Neutral Scene** (already balanced):
- Gains: `[1.0, 1.0, 1.0]` - No correction needed

## AWB Algorithms

There are several approaches to automatic white balance:

### 1. Gray World Assumption

**Principle**: The average color of a scene should be neutral gray.

```
If average scene color is yellow → reduce yellow
If average scene color is blue → reduce blue
```

**Pros**: Simple, works for many scenes
**Cons**: Fails with dominant colors (e.g., green grass, blue sky)

### 2. Face-Aware AWB

**Principle**: Prioritize accurate skin tones for portraits.

```
Focus on face regions
Compute AWB based on skin color
Ensure faces look natural
```

**Pros**: Excellent for portraits
**Cons**: Requires face detection

### 3. Segmentation-Based AWB

**Principle**: Different regions need different corrections.

```
Segment image into: sky, person, background
Apply region-specific AWB
Combine results intelligently
```

**Pros**: Handles complex scenes
**Cons**: Requires segmentation

### 4. Machine Learning AWB

**Principle**: Learn optimal corrections from training data.

```
Train neural network on many images
Learn patterns: lighting → correction
Apply learned model to new images
```

**Pros**: Can handle complex scenarios
**Cons**: Requires training data and model

## Our Implementation

This codebase implements a **hybrid approach** that combines multiple algorithms:

1. **Multiple Analyzers**: Run several AWB algorithms in parallel
2. **Trust Scores**: Each analyzer provides confidence level
3. **Intelligent Fusion**: Combine results weighted by confidence
4. **Temporal Smoothing**: For video, prevent flickering

### Why Hybrid?

Different algorithms work better in different scenarios:
- **Gray world**: Good for general scenes
- **Face-aware**: Best for portraits
- **Segmentation**: Handles complex scenes
- **ML**: Can learn from data

By combining them, we get the best of all approaches!

## Visual Example

### Before AWB
```
Warm indoor lighting:
- White wall appears yellow
- Skin tones look orange
- Overall image has warm cast
```

### After AWB
```
Corrected:
- White wall appears white
- Skin tones look natural
- Colors are balanced
```

See `results/awb_real_data_results/comparison.jpg` for a real example!

## Key Concepts Summary

1. **Color Temperature**: Describes the color of light (warm to cool)
2. **RGB Gains**: Multipliers to correct color channels
3. **Gray World**: Average scene should be neutral
4. **Face-Aware**: Prioritize skin tones
5. **Segmentation**: Region-specific corrections
6. **Hybrid Fusion**: Combine multiple approaches

## Next Steps

- **Tutorial**: `docs/TUTORIAL.md` - Step-by-step implementation
- **Algorithms**: `docs/ALGORITHMS.md` - Deep dive into each method
- **Examples**: `docs/EXAMPLES.md` - Code examples
- **API Reference**: `docs/API_REFERENCE.md` - Complete documentation

---

**Ready to code?** Start with `docs/TUTORIAL.md` to build your own AWB system!

