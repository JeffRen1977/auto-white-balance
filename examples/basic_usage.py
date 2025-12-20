"""
Basic AWB Usage Example

This example demonstrates the simplest way to use the AWB system.
Perfect for beginners to understand white balance correction.
"""

import sys
import os
import cv2
import numpy as np

# Add parent directory to path to import awb_system
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.awb_system import AutoWhiteBalanceSystem

def main():
    """Basic AWB example."""
    print("=" * 60)
    print("Basic Auto White Balance Example")
    print("=" * 60)
    
    # Initialize AWB system
    print("\n1. Initializing AWB system...")
    awb = AutoWhiteBalanceSystem()
    print("   ✓ AWB system ready")
    
    # Load an image (or create a test image)
    print("\n2. Loading image...")
    # For demonstration, create a test image with color cast
    # In practice, use: image = cv2.imread('your_image.jpg')
    
    # Create a warm (yellow/orange) test image
    image = np.ones((480, 640, 3), dtype=np.uint8) * 128
    image[:, :, 2] = image[:, :, 2] * 1.3  # Boost red (warm cast)
    image[:, :, 0] = image[:, :, 0] * 0.8  # Reduce blue
    
    print(f"   ✓ Image loaded: {image.shape[1]}x{image.shape[0]}")
    
    # Process with AWB
    print("\n3. Processing with AWB...")
    result = awb.process_frame(
        image,
        bv=5.0,        # Brightness value (from auto-exposure)
        scene="auto"   # Scene type
    )
    
    print(f"   ✓ Processing complete")
    print(f"   - AWB Gains: R={result.gains[0]:.3f}, G={result.gains[1]:.3f}, B={result.gains[2]:.3f}")
    print(f"   - Color Temperature: {result.cct:.1f}K")
    print(f"   - Confidence: {result.trust:.3f}")
    print(f"   - Processing Time: {result.processing_time:.2f}ms")
    
    # Apply AWB correction
    print("\n4. Applying AWB correction...")
    corrected = image.copy().astype(np.float32)
    corrected[:, :, 0] *= result.gains[2]  # Blue channel
    corrected[:, :, 1] *= result.gains[1]  # Green channel
    corrected[:, :, 2] *= result.gains[0]  # Red channel
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)
    
    print("   ✓ Correction applied")
    
    # Save results
    print("\n5. Saving results...")
    output_dir = "../results/basic_example"
    os.makedirs(output_dir, exist_ok=True)
    
    cv2.imwrite(f"{output_dir}/original.jpg", image)
    cv2.imwrite(f"{output_dir}/corrected.jpg", corrected)
    
    # Create side-by-side comparison
    comparison = np.hstack([image, corrected])
    cv2.imwrite(f"{output_dir}/comparison.jpg", comparison)
    
    print(f"   ✓ Results saved to {output_dir}/")
    print(f"     - original.jpg: Original image")
    print(f"     - corrected.jpg: AWB corrected image")
    print(f"     - comparison.jpg: Side-by-side comparison")
    
    print("\n" + "=" * 60)
    print("Example Complete!")
    print("=" * 60)
    print("\nKey Points:")
    print("  • AWB automatically detects color cast")
    print("  • Gains correct the color channels")
    print("  • Result: Natural, balanced colors")
    print("\nNext Steps:")
    print("  • Try with your own images")
    print("  • Read docs/INTRODUCTION.md for concepts")
    print("  • Read docs/TUTORIAL.md for step-by-step guide")

if __name__ == "__main__":
    main()

