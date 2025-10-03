"""
Auto White Balance System Demo

This script demonstrates the AWB system with various test scenarios
including different lighting conditions, scene types, and analyzer configurations.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from awb_system import (
    AutoWhiteBalanceSystem,
    HeuristicAnalyzer,
    FaceAwareAnalyzer,
    SegmentationAnalyzer,
    MLAnalyzer,
    FrameStatsExtractor,
    create_test_image
)

def create_synthetic_scene(width=640, height=480, scene_type="portrait", 
                          color_bias="warm", add_face=True):
    """Create synthetic test scenes with known characteristics."""
    
    # Create base image
    image = np.ones((height, width, 3), dtype=np.float32) * 128
    
    if scene_type == "portrait":
        # Create portrait scene with skin tone
        # Face region
        face_center_x, face_center_y = width // 2, height // 2 - 50
        face_size = 100
        
        # Create face mask (simple ellipse)
        y, x = np.ogrid[:height, :width]
        face_mask = ((x - face_center_x) / (face_size // 2)) ** 2 + \
                   ((y - face_center_y) / (face_size // 1.3)) ** 2 <= 1
        
        # Apply skin tone (warm colors)
        image[face_mask] = [180, 150, 120]  # BGR format
        
        # Background
        image[~face_mask] = [100, 120, 140]
        
        face_rois = [(face_center_x - face_size//2, face_center_y - face_size//2, 
                     face_size, face_size)] if add_face else []
        
    elif scene_type == "landscape":
        # Create landscape scene
        # Sky (top third)
        sky_region = slice(0, height//3)
        image[sky_region] = [200, 180, 150]  # Blue sky
        
        # Ground (bottom two thirds)
        ground_region = slice(height//3, height)
        image[ground_region] = [80, 100, 60]  # Green ground
        
        face_rois = []
        
    elif scene_type == "indoor":
        # Create indoor scene
        image[:, :] = [120, 110, 100]  # Indoor lighting
        
        # Add some objects
        cv2.rectangle(image, (100, 200, 80, 120), (200, 180, 150), -1)  # Object 1
        cv2.rectangle(image, (400, 300, 60, 80), (180, 160, 140), -1)   # Object 2
        
        face_rois = []
        
    else:  # neutral
        image[:, :] = [128, 128, 128]  # Neutral gray
        face_rois = []
    
    # Apply color bias
    if color_bias == "warm":
        image[:, :, 2] *= 1.3  # Boost red
        image[:, :, 0] *= 0.8  # Reduce blue
    elif color_bias == "cool":
        image[:, :, 0] *= 1.3  # Boost blue
        image[:, :, 2] *= 0.8  # Reduce red
    
    # Add noise
    noise = np.random.normal(0, 8, image.shape)
    image = np.clip(image + noise, 0, 255)
    
    return image.astype(np.uint8), face_rois

def create_segmentation_map(width=640, height=480, scene_type="portrait"):
    """Create synthetic segmentation map."""
    # Resize to grid size (64x48 by default)
    grid_width, grid_height = 64, 48
    
    seg_map = np.zeros((grid_height, grid_width), dtype=np.int32)
    
    if scene_type == "portrait":
        # Face region (segment 1)
        face_center_x, face_center_y = grid_width // 2, grid_height // 2
        face_size = 15
        
        y, x = np.ogrid[:grid_height, :grid_width]
        face_mask = ((x - face_center_x) / (face_size // 2)) ** 2 + \
                   ((y - face_center_y) / (face_size // 1.3)) ** 2 <= 1
        seg_map[face_mask] = 1  # Person/face
        
        # Background
        seg_map[seg_map == 0] = 0  # Background
        
    elif scene_type == "landscape":
        # Sky (top third)
        seg_map[:grid_height//3] = 2  # Sky
        # Ground (bottom two thirds)
        seg_map[grid_height//3:] = 0  # Ground/background
        
    else:
        # All background
        seg_map[:, :] = 0
    
    return seg_map

def demonstrate_basic_awb():
    """Demonstrate basic AWB functionality."""
    print("=" * 60)
    print("BASIC AWB DEMONSTRATION")
    print("=" * 60)
    
    # Create test images
    warm_portrait, face_rois = create_synthetic_scene(scene_type="portrait", color_bias="warm")
    cool_landscape, _ = create_synthetic_scene(scene_type="landscape", color_bias="cool")
    
    # Initialize AWB system
    awb_system = AutoWhiteBalanceSystem()
    
    # Process warm portrait
    print("\n1. Processing Warm Portrait:")
    print("-" * 30)
    warm_result = awb_system.process_frame(
        warm_portrait,
        bv=5.0,
        scene="portrait",
        face_rois=face_rois
    )
    
    print(f"   Gains: R={warm_result.gains[0]:.3f}, G={warm_result.gains[1]:.3f}, B={warm_result.gains[2]:.3f}")
    print(f"   CCT: {warm_result.cct:.1f}K")
    print(f"   Trust: {warm_result.trust:.3f}")
    print(f"   Processing time: {warm_result.processing_time:.2f}ms")
    
    # Process cool landscape
    print("\n2. Processing Cool Landscape:")
    print("-" * 30)
    cool_result = awb_system.process_frame(
        cool_landscape,
        bv=7.0,
        scene="landscape"
    )
    
    print(f"   Gains: R={cool_result.gains[0]:.3f}, G={cool_result.gains[1]:.3f}, B={cool_result.gains[2]:.3f}")
    print(f"   CCT: {cool_result.cct:.1f}K")
    print(f"   Trust: {cool_result.trust:.3f}")
    print(f"   Processing time: {cool_result.processing_time:.2f}ms")

def demonstrate_analyzer_comparison():
    """Compare different analyzers."""
    print("\n" + "=" * 60)
    print("ANALYZER COMPARISON")
    print("=" * 60)
    
    # Create test image
    test_image, face_rois = create_synthetic_scene(scene_type="portrait", color_bias="warm")
    seg_map = create_segmentation_map(scene_type="portrait")
    
    # Create individual analyzers
    analyzers = {
        "Heuristic": HeuristicAnalyzer(),
        "Face-Aware": FaceAwareAnalyzer(),
        "Segmentation": SegmentationAnalyzer(),
        "ML": MLAnalyzer()
    }
    
    # Extract frame summary
    extractor = FrameStatsExtractor()
    frame = extractor.extract_from_image(
        test_image,
        bv=5.0,
        scene="portrait",
        face_rois=face_rois,
        segmentation_map=seg_map
    )
    
    print(f"\nAnalyzing test image with warm color bias...")
    print("-" * 50)
    
    for name, analyzer in analyzers.items():
        result = analyzer.analyze(frame)
        print(f"{name:12}: Gains=[{result.gains[0]:.3f}, {result.gains[1]:.3f}, {result.gains[2]:.3f}], "
              f"CCT={result.cct:.1f}K, Trust={result.trust:.3f}")

def demonstrate_temporal_filtering():
    """Demonstrate temporal filtering effects."""
    print("\n" + "=" * 60)
    print("TEMPORAL FILTERING DEMONSTRATION")
    print("=" * 60)
    
    # Create sequence of images with changing lighting
    images = []
    for i in range(5):
        # Gradually change from warm to cool
        bias_strength = 1.0 - (i / 4.0)  # From 1.0 to 0.0
        if bias_strength > 0.5:
            color_bias = "warm"
        else:
            color_bias = "cool"
        
        image, face_rois = create_synthetic_scene(
            scene_type="portrait", 
            color_bias=color_bias
        )
        images.append((image, face_rois))
    
    # Test with and without temporal filtering
    print("\n1. Without Temporal Filtering (alpha=1.0):")
    print("-" * 45)
    awb_no_filter = AutoWhiteBalanceSystem()
    awb_no_filter.temporal_filter.alpha_gains = 1.0  # No filtering
    
    for i, (image, face_rois) in enumerate(images):
        result = awb_no_filter.process_frame(image, bv=5.0, scene="portrait", face_rois=face_rois)
        print(f"Frame {i+1}: Gains=[{result.gains[0]:.3f}, {result.gains[1]:.3f}, {result.gains[2]:.3f}], "
              f"CCT={result.cct:.1f}K")
    
    print("\n2. With Temporal Filtering (alpha=0.3):")
    print("-" * 42)
    awb_with_filter = AutoWhiteBalanceSystem()
    awb_with_filter.temporal_filter.alpha_gains = 0.3  # Strong filtering
    
    for i, (image, face_rois) in enumerate(images):
        result = awb_with_filter.process_frame(image, bv=5.0, scene="portrait", face_rois=face_rois)
        print(f"Frame {i+1}: Gains=[{result.gains[0]:.3f}, {result.gains[1]:.3f}, {result.gains[2]:.3f}], "
              f"CCT={result.cct:.1f}K")

def demonstrate_scene_adaptation():
    """Demonstrate scene-specific adaptations."""
    print("\n" + "=" * 60)
    print("SCENE ADAPTATION DEMONSTRATION")
    print("=" * 60)
    
    scenes = [
        ("portrait", "Portrait Scene"),
        ("landscape", "Landscape Scene"),
        ("indoor", "Indoor Scene")
    ]
    
    awb_system = AutoWhiteBalanceSystem()
    
    for scene_type, scene_name in scenes:
        print(f"\n{scene_name}:")
        print("-" * 20)
        
        # Create scene with neutral lighting
        image, face_rois = create_synthetic_scene(scene_type=scene_type, color_bias="neutral")
        seg_map = create_segmentation_map(scene_type=scene_type)
        
        result = awb_system.process_frame(
            image,
            bv=6.0,
            scene=scene_type,
            face_rois=face_rois,
            segmentation_map=seg_map
        )
        
        print(f"   Gains: R={result.gains[0]:.3f}, G={result.gains[1]:.3f}, B={result.gains[2]:.3f}")
        print(f"   CCT: {result.cct:.1f}K")
        print(f"   Trust: {result.trust:.3f}")
        
        # Show analyzer contributions
        print(f"   Analyzer Results:")
        for analyzer_result in result.analyzer_results:
            print(f"     {analyzer_result.tag}: trust={analyzer_result.trust:.3f}, cct={analyzer_result.cct:.1f}K")

def demonstrate_performance():
    """Demonstrate performance characteristics."""
    print("\n" + "=" * 60)
    print("PERFORMANCE DEMONSTRATION")
    print("=" * 60)
    
    # Create test image
    test_image, face_rois = create_synthetic_scene()
    
    # Test different configurations
    configurations = [
        ("Minimal", [HeuristicAnalyzer()]),
        ("Standard", [HeuristicAnalyzer(), FaceAwareAnalyzer()]),
        ("Full", [HeuristicAnalyzer(), FaceAwareAnalyzer(), SegmentationAnalyzer(), MLAnalyzer()])
    ]
    
    for config_name, analyzers in configurations:
        print(f"\n{config_name} Configuration:")
        print("-" * 25)
        
        awb_system = AutoWhiteBalanceSystem(analyzers=analyzers)
        
        # Process multiple frames
        for i in range(10):
            result = awb_system.process_frame(
                test_image,
                bv=5.0 + np.random.normal(0, 0.5),  # Add some variation
                scene="portrait",
                face_rois=face_rois
            )
        
        # Get performance stats
        stats = awb_system.get_performance_stats()
        print(f"   Average processing time: {stats['average_time']:.2f}ms")
        print(f"   Estimated FPS: {stats['fps_estimate']:.1f}")
        print(f"   Analyzers used: {len(analyzers)}")

def save_results_visualization():
    """Save visualization of AWB results."""
    print("\n" + "=" * 60)
    print("SAVING VISUALIZATION RESULTS")
    print("=" * 60)
    
    # Create test images
    scenes = ["portrait", "landscape", "indoor"]
    biases = ["warm", "cool", "neutral"]
    
    awb_system = AutoWhiteBalanceSystem()
    
    for scene in scenes:
        for bias in biases:
            # Create test image
            image, face_rois = create_synthetic_scene(scene_type=scene, color_bias=bias)
            
            # Process with AWB
            result = awb_system.process_frame(
                image,
                bv=6.0,
                scene=scene,
                face_rois=face_rois
            )
            
            # Apply AWB correction
            corrected_image = image.copy().astype(np.float32)
            corrected_image[:, :, 0] *= result.gains[2]  # Blue
            corrected_image[:, :, 1] *= result.gains[1]  # Green  
            corrected_image[:, :, 2] *= result.gains[0]  # Red
            corrected_image = np.clip(corrected_image, 0, 255).astype(np.uint8)
            
            # Save results
            filename = f"awb_demo_{scene}_{bias}"
            cv2.imwrite(f"{filename}_original.jpg", image)
            cv2.imwrite(f"{filename}_corrected.jpg", corrected_image)
            
            print(f"Saved: {filename}_original.jpg, {filename}_corrected.jpg")
            print(f"  Original gains: [1.000, 1.000, 1.000]")
            print(f"  AWB gains: [{result.gains[0]:.3f}, {result.gains[1]:.3f}, {result.gains[2]:.3f}]")
            print(f"  CCT: {result.cct:.1f}K, Trust: {result.trust:.3f}")

def main():
    """Run all demonstrations."""
    print("Auto White Balance System - Comprehensive Demo")
    print("=" * 60)
    
    try:
        # Run demonstrations
        demonstrate_basic_awb()
        demonstrate_analyzer_comparison()
        demonstrate_temporal_filtering()
        demonstrate_scene_adaptation()
        demonstrate_performance()
        save_results_visualization()
        
        print("\n" + "=" * 60)
        print("DEMO COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print("\nGenerated files:")
        print("- awb_demo_*_original.jpg: Original test images")
        print("- awb_demo_*_corrected.jpg: AWB-corrected images")
        print("\nThe demo shows various aspects of the AWB system:")
        print("1. Basic functionality with different scene types")
        print("2. Comparison of different analyzers")
        print("3. Temporal filtering effects")
        print("4. Scene-specific adaptations")
        print("5. Performance characteristics")
        print("6. Visual results comparison")
        
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
