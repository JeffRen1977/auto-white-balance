"""
Test Auto White Balance System with Real RAW Data

This script processes real RAW10 sensor data and applies AWB corrections.
The RAW file contains: RAW10_4032x756 sensor data
"""

import sys
import os
import numpy as np
import cv2
import json

# Add parent directory to path to import awb_system
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.awb_system import (
    AutoWhiteBalanceSystem,
    FrameStatsExtractor,
    HeuristicAnalyzer,
    FaceAwareAnalyzer,
    SegmentationAnalyzer,
    MLAnalyzer,
    FrameSummary
)

def parse_raw_filename(filename):
    """Parse RAW filename to extract metadata."""
    # Format: YYYYMMDD_HHMMSS_MS_FORMAT_WIDTHxHEIGHT_RS_ROWSTRIDE_...
    parts = filename.split('_')
    
    metadata = {
        'filename': filename,
        'date': parts[0] if len(parts) > 0 else None,
        'time': parts[1] if len(parts) > 1 else None,
        'format': 'RAW10',
        'width': 4032,
        'height': 756,
        'row_stride': 5056,
        'pixel_stride': 1,
        'camera_id': 3,
    }
    
    # Try to parse dimensions from filename
    for i, part in enumerate(parts):
        if 'x' in part and part.replace('x', '').replace('RAW10_', '').isdigit():
            # Found dimension string like "4032x3024" or "RAW10_4032x3024"
            dim_str = part.replace('RAW10_', '')
            try:
                w, h = dim_str.split('x')
                metadata['width'] = int(w)
                metadata['height'] = int(h)
            except:
                pass
        
        # Parse row stride
        if part == 'RS' and i + 1 < len(parts):
            try:
                metadata['row_stride'] = int(parts[i + 1])
            except:
                pass
        
        # Parse camera ID
        if part == 'cam' and i + 1 < len(parts):
            try:
                metadata['camera_id'] = int(parts[i + 1])
            except:
                pass
    
    return metadata

def read_raw10_data(filepath, width, height, row_stride=None):
    """
    Read RAW10 format data with proper row stride handling.
    
    RAW10 format: 4 pixels packed into 5 bytes
    Each 4 pixels use 5 bytes: [P0_high][P1_high][P2_high][P3_high][P0P1P2P3_low]
    
    Row Stride: The actual bytes allocated per row (may include padding)
                If row_stride > bytes_per_row, there are padding bytes to skip
    """
    print(f"\nReading RAW10 data from: {filepath}")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"RAW file not found: {filepath}")
    
    file_size = os.path.getsize(filepath)
    print(f"File size: {file_size} bytes")
    
    # Calculate bytes needed per row for RAW10 format
    # RAW10: 4 pixels packed into 5 bytes, so width pixels = (width * 10 + 7) // 8 bytes
    bytes_per_row = (width * 10 + 7) // 8  # Ceiling division for packed RAW10
    
    # Row stride: actual bytes allocated per row (includes padding if any)
    # In camera systems, row_stride is typically in bytes and may include alignment/padding
    if row_stride is None:
        row_stride_bytes = bytes_per_row  # No padding if not specified
    else:
        # row_stride from metadata is typically in bytes already
        # If it seems too small, it might be in pixels (convert to bytes)
        if row_stride < width:
            # Assume it's in pixels, convert to bytes
            row_stride_bytes = (row_stride * 10 + 7) // 8
        elif row_stride < bytes_per_row:
            # Too small to be valid, use calculated value
            row_stride_bytes = bytes_per_row
        else:
            # row_stride is in bytes (most common case)
            row_stride_bytes = row_stride
    
    row_stride = row_stride_bytes
    
    padding_per_row = row_stride - bytes_per_row
    
    print(f"Image dimensions: {width}x{height}")
    print(f"Row stride: {row_stride} bytes")
    print(f"Bytes per row (RAW10 packed): {bytes_per_row} bytes")
    print(f"Padding per row: {padding_per_row} bytes")
    
    # Read raw bytes
    with open(filepath, 'rb') as f:
        raw_bytes = np.frombuffer(f.read(), dtype=np.uint8)
    
    print(f"Actual data size: {len(raw_bytes)} bytes")
    print(f"Expected with stride: {row_stride * height} bytes")
    
    # Initialize output array (height x width)
    image_data = np.zeros((height, width), dtype=np.uint16)
    
    # Process each row separately to handle row stride padding
    byte_offset = 0
    
    for row in range(height):
        if byte_offset + bytes_per_row > len(raw_bytes):
            print(f"Warning: Row {row} exceeds file size")
            break
        
        # Extract one row's data (skip padding if any)
        row_bytes = raw_bytes[byte_offset:byte_offset + bytes_per_row]
        
        # Unpack this row: process in chunks of 5 bytes -> 4 pixels
        pixels_in_row = []
        for i in range(0, len(row_bytes) - 4, 5):  # Step by 5 bytes
            if i + 4 >= len(row_bytes):
                break
            
            b0, b1, b2, b3, b4 = row_bytes[i], row_bytes[i+1], row_bytes[i+2], row_bytes[i+3], row_bytes[i+4]
            
            # Reconstruct 4 10-bit pixels
            # Each pixel = 8 high bits + 2 low bits from b4
            pixels_in_row.append((b0 << 2) | ((b4 >> 0) & 0x03))  # Pixel 0
            pixels_in_row.append((b1 << 2) | ((b4 >> 2) & 0x03))  # Pixel 1
            pixels_in_row.append((b2 << 2) | ((b4 >> 4) & 0x03))  # Pixel 2
            pixels_in_row.append((b3 << 2) | ((b4 >> 6) & 0x03))  # Pixel 3
        
        # Store pixels for this row (only width pixels)
        num_pixels = min(len(pixels_in_row), width)
        image_data[row, :num_pixels] = pixels_in_row[:num_pixels]
        
        # Move to next row (skip padding if any)
        byte_offset += row_stride
    
    print(f"Successfully unpacked {height} rows")
    print(f"Image shape: {image_data.shape}")
    print(f"Pixel value range: [{image_data.min()}, {image_data.max()}]")
    print(f"Mean pixel value: {image_data.mean():.2f}")
    
    return image_data

def debayer_image(raw_image, pattern='RGGB'):
    """
    Debayer (demosaic) Bayer pattern data using OpenCV.
    
    Args:
        raw_image: Raw sensor data (H x W, uint16, typically 0-1023 for 10-bit)
        pattern: Bayer pattern ('RGGB', 'BGGR', 'GRBG', 'GBRG')
                 RGGB pattern layout:
                 Row 0: R G R G R G ...
                 Row 1: G B G B G B ...
    
    Returns:
        RGB image (H x W x 3, float32, 0-1023 range)
    """
    # Detect actual bit depth from image data
    max_val = raw_image.max()
    if max_val > 4095:
        bit_depth = 16
        max_value = 65535
    elif max_val > 1023:
        bit_depth = 12
        max_value = 4095
    else:
        bit_depth = 10
        max_value = 1023
    
    print(f"Detected bit depth: {bit_depth}-bit (max value: {max_val})")
    
    # Normalize to 8-bit for OpenCV debayering
    # OpenCV's debayering functions work on 8-bit images
    raw_8bit = np.clip((raw_image.astype(np.float32) / max_value * 255.0), 0, 255).astype(np.uint8)
    
    # OpenCV color code mapping:
    # COLOR_BayerXX2BGR means: input pattern is XX, output is BGR
    # So for RGGB input, we use COLOR_BayerRG2BGR
    pattern_to_code = {
        'RGGB': cv2.COLOR_BayerRG2BGR,  # RGGB input -> BGR output
        'BGGR': cv2.COLOR_BayerBG2BGR,  # BGGR input -> BGR output
        'GRBG': cv2.COLOR_BayerGR2BGR,  # GRBG input -> BGR output
        'GBRG': cv2.COLOR_BayerGB2BGR,  # GBRG input -> BGR output
    }
    
    color_code = pattern_to_code.get(pattern, cv2.COLOR_BayerRG2BGR)
    
    # Debayer using OpenCV (outputs BGR format)
    bgr_image = cv2.cvtColor(raw_8bit, color_code)
    
    # Convert back to original bit depth range and RGB format
    # OpenCV outputs BGR, we need RGB, so reverse channels
    rgb_image = bgr_image[:, :, ::-1].astype(np.float32)  # BGR -> RGB
    rgb_image = rgb_image / 255.0 * max_value  # Scale back to original range
    
    return rgb_image

def extract_bayer_statistics(raw_image, pattern='RGGB', grid_size=(64, 48)):
    """
    Extract Bayer RGB statistics directly from RAW data.
    This is more efficient than debayering the full image.
    """
    height, width = raw_image.shape
    grid_h, grid_w = grid_size
    
    # Initialize grids
    r_grid = np.zeros((grid_h, grid_w), dtype=np.float32)
    g_grid = np.zeros((grid_h, grid_w), dtype=np.float32)
    b_grid = np.zeros((grid_h, grid_w), dtype=np.float32)
    
    # Calculate block sizes
    block_h = height // grid_h
    block_w = width // grid_w
    
    for i in range(grid_h):
        for j in range(grid_w):
            # Extract block
            y_start = i * block_h
            y_end = min((i + 1) * block_h, height)
            x_start = j * block_w
            x_end = min((j + 1) * block_w, width)
            
            block = raw_image[y_start:y_end, x_start:x_end]
            
            if pattern == 'RGGB':
                # Extract R, G, B values based on Bayer pattern
                r_vals = block[0::2, 0::2]  # Red pixels
                g_vals1 = block[0::2, 1::2]  # Green pixels (red row)
                g_vals2 = block[1::2, 0::2]  # Green pixels (blue row)
                b_vals = block[1::2, 1::2]   # Blue pixels
                
                r_grid[i, j] = np.mean(r_vals)
                g_grid[i, j] = np.mean(np.concatenate([g_vals1.flatten(), g_vals2.flatten()]))
                b_grid[i, j] = np.mean(b_vals)
    
    return r_grid, g_grid, b_grid

def estimate_brightness_value(raw_image):
    """Estimate brightness value (BV) from raw image."""
    mean_intensity = np.mean(raw_image)
    max_intensity = 1023.0  # 10-bit max
    
    # Simple BV estimation: log2 based
    normalized_intensity = mean_intensity / max_intensity
    
    # BV typically ranges from -6 (very dark) to +15 (very bright)
    # Rough approximation
    if normalized_intensity > 0:
        bv = np.log2(normalized_intensity) + 10
    else:
        bv = -6
    
    return np.clip(bv, -6, 15)

def apply_awb_to_raw(raw_image, gains, pattern='RGGB'):
    """Apply AWB gains directly to RAW Bayer data."""
    corrected = raw_image.copy().astype(np.float32)
    
    r_gain, g_gain, b_gain = gains
    
    if pattern == 'RGGB':
        # Apply gains to respective Bayer positions
        corrected[0::2, 0::2] *= r_gain  # Red
        corrected[0::2, 1::2] *= g_gain  # Green (red row)
        corrected[1::2, 0::2] *= g_gain  # Green (blue row)
        corrected[1::2, 1::2] *= b_gain  # Blue
    
    # Clip to valid range
    corrected = np.clip(corrected, 0, 1023)
    
    return corrected.astype(np.uint16)

def save_rgb_image(rgb_image, filename, bit_depth=10):
    """
    Save RGB image to file with proper normalization.
    
    Args:
        rgb_image: RGB image (H x W x 3, float32)
        filename: Output filename
        bit_depth: Bit depth for normalization (default: 10 for 10-bit images)
    """
    # Detect actual max value from image data
    img_max = rgb_image.max()
    if img_max > 1023:
        # Auto-detect bit depth from actual data
        if img_max > 4095:
            max_value = 65535  # 16-bit
        else:
            max_value = 4095   # 12-bit
    else:
        max_value = 1023       # 10-bit
    
    # Normalize to 8-bit for JPEG display
    # Ensure values are in valid range
    rgb_normalized = np.clip(rgb_image, 0, max_value)
    rgb_8bit = (rgb_normalized / max_value * 255.0).astype(np.uint8)
    
    # OpenCV uses BGR format for saving, so convert RGB -> BGR
    bgr_8bit = rgb_8bit[:, :, ::-1]  # Reverse channels: RGB -> BGR
    
    cv2.imwrite(filename, bgr_8bit)
    
    # Print statistics for verification
    print(f"Saved: {filename}")
    print(f"  Image shape: {rgb_image.shape}")
    print(f"  Value range: [{rgb_image.min():.1f}, {rgb_image.max():.1f}]")
    print(f"  Normalized range: [{rgb_8bit.min()}, {rgb_8bit.max()}]")

def main():
    """
    Main test function for real RAW data.
    
    This function processes RAW10 sensor data files from the data/ directory.
    
    Expected Directory Structure:
        AutoWhiteBalance/
        ├── data/                          # RAW files stored here
        │   └── *.raw                     # RAW10 sensor data files
        ├── tests/
        │   └── test_real_data.py         # This script
        └── results/
            └── awb_real_data_results/    # Output saved here
    
    The script will:
    1. Look for RAW files in data/ directory
    2. Process the newest RAW file found
    3. Apply AWB correction
    4. Save results to results/awb_real_data_results/
    """
    print("=" * 70)
    print("AUTO WHITE BALANCE SYSTEM - REAL DATA TEST")
    print("=" * 70)
    
    # Find all RAW files in data directory
    # Path structure: AutoWhiteBalance/tests/test_real_data.py -> AutoWhiteBalance/data/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # Go up from tests/ to project root
    data_dir = os.path.join(project_root, 'data')
    data_dir = os.path.abspath(data_dir)
    
    print(f"\nLooking for RAW files in: {data_dir}")
    
    if not os.path.exists(data_dir):
        print(f"ERROR: Data directory not found: {data_dir}")
        print(f"Please create the data/ directory and place RAW files there")
        return
    
    if not os.path.isdir(data_dir):
        print(f"ERROR: {data_dir} exists but is not a directory")
        return
    
    raw_files = [f for f in os.listdir(data_dir) if f.endswith('.raw')]
    
    if not raw_files:
        print(f"ERROR: No RAW files found in {data_dir}")
        print(f"Please ensure RAW files are placed in the data/ directory")
        return
    
    # Sort by modification time (newest first)
    raw_files.sort(key=lambda f: os.path.getmtime(os.path.join(data_dir, f)), reverse=True)
    
    print(f"\nFound {len(raw_files)} RAW file(s) in {data_dir}:")
    for i, f in enumerate(raw_files, 1):
        size_mb = os.path.getsize(os.path.join(data_dir, f)) / (1024 * 1024)
        print(f"  {i}. {f} ({size_mb:.1f} MB)")
    
    # Use the newest/first file
    raw_filename = raw_files[0]
    raw_filepath = os.path.join(data_dir, raw_filename)
    
    print(f"\nUsing: {raw_filename}")
    
    if not os.path.exists(raw_filepath):
        print(f"ERROR: RAW file not found: {raw_filepath}")
        return
    
    # Parse metadata
    metadata = parse_raw_filename(raw_filename)
    print("\nRAW File Metadata:")
    print("-" * 40)
    for key, value in metadata.items():
        print(f"  {key}: {value}")
    
    # Read RAW data
    try:
        raw_image = read_raw10_data(
            raw_filepath, 
            metadata['width'], 
            metadata['height'],
            metadata['row_stride']
        )
    except Exception as e:
        print(f"\nERROR reading RAW data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Estimate brightness
    bv = estimate_brightness_value(raw_image)
    print(f"\nEstimated Brightness Value (BV): {bv:.2f}")
    
    # Extract Bayer statistics (efficient method)
    print("\nExtracting Bayer RGB statistics...")
    r_grid, g_grid, b_grid = extract_bayer_statistics(raw_image, pattern='RGGB')
    print(f"Statistics grid shape: {r_grid.shape}")
    print(f"R mean: {np.mean(r_grid):.2f}, G mean: {np.mean(g_grid):.2f}, B mean: {np.mean(b_grid):.2f}")
    
    # Create frame summary
    frame = FrameSummary(
        r_grid=r_grid,
        g_grid=g_grid,
        b_grid=b_grid,
        bv=bv,
        scene="auto",
        face_rois=None,
        segmentation_map=None,
        spectral_data=None,
        depth_info=None
    )
    
    # Initialize AWB system
    print("\nInitializing AWB System...")
    awb_system = AutoWhiteBalanceSystem()
    
    # Process frame
    print("\nProcessing frame with AWB...")
    result = awb_system.process_frame(frame)
    
    # Display results
    print("\n" + "=" * 70)
    print("AWB RESULTS")
    print("=" * 70)
    print(f"\nAWB Gains:")
    print(f"  R gain: {result.gains[0]:.4f}")
    print(f"  G gain: {result.gains[1]:.4f}")
    print(f"  B gain: {result.gains[2]:.4f}")
    print(f"\nColor Temperature: {result.cct:.1f}K")
    print(f"Confidence (Trust): {result.trust:.3f}")
    print(f"Processing Time: {result.processing_time:.2f}ms")
    
    print(f"\nColor Correction Matrix (CCM):")
    print(result.ccm)
    
    print(f"\nIndividual Analyzer Results:")
    print("-" * 50)
    for analyzer_result in result.analyzer_results:
        print(f"\n{analyzer_result.tag.upper()}:")
        print(f"  Gains: [{analyzer_result.gains[0]:.4f}, {analyzer_result.gains[1]:.4f}, {analyzer_result.gains[2]:.4f}]")
        print(f"  CCT: {analyzer_result.cct:.1f}K")
        print(f"  Trust: {analyzer_result.trust:.3f}")
        if analyzer_result.metadata:
            print(f"  Metadata: {analyzer_result.metadata}")
    
    # Apply AWB to raw data
    print("\n" + "=" * 70)
    print("APPLYING AWB CORRECTIONS")
    print("=" * 70)
    
    print("\nApplying AWB gains to RAW data...")
    corrected_raw = apply_awb_to_raw(raw_image, result.gains, pattern='RGGB')
    
    # Debayer original and corrected images
    print("\nDebayering original image...")
    original_rgb = debayer_image(raw_image, pattern='RGGB')
    
    print("Debayering corrected image...")
    corrected_rgb = debayer_image(corrected_raw, pattern='RGGB')
    
    # Save results to results directory
    # Path structure: AutoWhiteBalance/tests/ -> AutoWhiteBalance/results/awb_real_data_results/
    output_dir = os.path.join(project_root, 'results', 'awb_real_data_results')
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nSaving results to {output_dir}/...")
    save_rgb_image(original_rgb, f"{output_dir}/original_image.jpg", bit_depth=10)
    save_rgb_image(corrected_rgb, f"{output_dir}/awb_corrected_image.jpg", bit_depth=10)
    
    # Helper function to convert numpy types to Python types
    def convert_to_python_types(obj):
        """Recursively convert numpy types to Python types for JSON serialization."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_to_python_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_python_types(item) for item in obj]
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        else:
            return obj
    
    # Save metadata
    result_metadata = {
        'raw_file': raw_filename,
        'image_dimensions': {'width': metadata['width'], 'height': metadata['height']},
        'brightness_value': float(bv),
        'awb_gains': {
            'r_gain': float(result.gains[0]),
            'g_gain': float(result.gains[1]),
            'b_gain': float(result.gains[2])
        },
        'color_temperature': float(result.cct),
        'confidence': float(result.trust),
        'processing_time_ms': float(result.processing_time),
        'ccm': result.ccm.tolist(),
        'analyzer_results': [
            {
                'analyzer': ar.tag,
                'gains': ar.gains.tolist(),
                'cct': float(ar.cct),
                'trust': float(ar.trust),
                'metadata': convert_to_python_types(ar.metadata)
            }
            for ar in result.analyzer_results
        ]
    }
    
    metadata_file = f"{output_dir}/awb_results.json"
    with open(metadata_file, 'w') as f:
        json.dump(result_metadata, f, indent=2)
    print(f"Saved: {metadata_file}")
    
    # Create comparison visualization
    print("\nCreating comparison visualization...")
    comparison = np.hstack([
        (original_rgb / 1023 * 255).astype(np.uint8),
        (corrected_rgb / 1023 * 255).astype(np.uint8)
    ])
    cv2.imwrite(f"{output_dir}/comparison.jpg", comparison)
    print(f"Saved: {output_dir}/comparison.jpg")
    
    print("\n" + "=" * 70)
    print("TEST COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"\nResults saved in: {output_dir}/")
    print("  - original_image.jpg: Original debayered image")
    print("  - awb_corrected_image.jpg: AWB corrected image")
    print("  - comparison.jpg: Side-by-side comparison")
    print("  - awb_results.json: Detailed AWB results and metadata")
    print("\nThe real data test demonstrates:")
    print("  1. RAW10 format parsing and unpacking")
    print("  2. Bayer pattern statistics extraction")
    print("  3. AWB processing on real sensor data")
    print("  4. Brightness estimation from raw data")
    print("  5. Direct AWB application to Bayer data")
    print("  6. Debayering and visualization")

if __name__ == "__main__":
    main()

