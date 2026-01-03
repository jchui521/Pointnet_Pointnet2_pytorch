"""
Twinner01 Data Preparation Script
Converts annotated .xyz files (similar to S3DIS) into training format

Expected Input Structure:
    raw_data/
        scene_01/
            object_type_1_001.txt    # X Y Z R G B (space-separated)
            object_type_1_002.txt
            object_type_2_001.txt
            background_001.txt
            ...
        scene_02/
            ...

Each .txt file contains points belonging to one object instance.
Filename prefix determines the class (e.g., "chair_01.txt" → "chair" class)
"""

import numpy as np
import os
import glob
from pathlib import Path
from tqdm import tqdm

def collect_scene_from_annotations(anno_dir, output_file, class_mapping):
    """
    Collect all annotated .txt files from a scene directory and combine them
    Similar to S3DIS processing
    
    Argsanno_dir: Directory containing annotated .txt files (one per object instance)
        output_file: Path to save combined scene as .npy
        class_mapping: Dictionary mapping class names to label integers
    
    Each .txt file format: X Y Z R G B (space or comma separated)
    Filename format: <class_name>_<instance_id>.txt
    Example: chair_01.txt, table_02.txt, floor_01.txt
    """
    anno_path = Path(anno_dir)
    
    if not anno_path.exists():
        raise ValueError(f"Annotation directory not found: {anno_dir}")
    
    # Find all .txt files
    txt_files = list(anno_path.glob('*.txt'))
    
    if len(txt_files) == 0:
        raise ValueError(f"No .txt files found in {anno_dir}")
    
    print(f"Processing {len(txt_files)} annotation files from {anno_dir}")
    
    all_points = []
    unknown_classes = set()
    
    for txt_file in txt_files:
        # Extract class name from filename
        # e.g., "chair_01.txt" -> "chair"
        filename = txt_file.stem
        class_name = filename.rsplit('_', 1)[0] if '_' in filename else filename
        
        # Get label ID
        if class_name not in class_mapping:
            if class_name not in unknown_classes:
                print(f"Warning: Unknown class '{class_name}', assigning to 'background' (label 0)")
                unknown_classes.add(class_name)
            label_id = 0  # Default to background
        else:
            label_id = class_mapping[class_name]
        
        # Load points from file
        try:
            # Try space-separated
            points = np.loadtxt(txt_file)
            
            if points.ndim == 1:
                points = points.reshape(1, -1)
            
            # Validate format
            if points.shape[1] == 6:
                # X Y Z R G B format
                xyz = points[:, :3]
                rgb = points[:, 3:6]
            elif points.shape[1] == 3:
                # Only X Y Z, add default gray color
                xyz = points
                rgb = np.ones((points.shape[0], 3)) * 128
            else:
                print(f"Warning: Unexpected format in {txt_file}, shape {points.shape}, skipping")
                continue
            
            # Add labels
            labels = np.ones((xyz.shape[0], 1)) * label_id
            scenes(input_dir, output_dir, class_mapping):
    """
    Process all scene directories (S3DIS-style structure)
    
    Args:
        input_dir: Root directory containing scene subdirectories
        output_dir: Directory to save processed .npy files
        class_mapping: Dictionary mapping class names to label integers
    
    Expected structure:
        input_dir/
            scene_01/
                object_01.txt
                object_02.txt
                ...
            scene_02/
                ...
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all scene directories
    scene_dirs = [d for d in input_path.iterdir() if d.is_dir()]
    
    if len(scene_dirs) == 0:
        raise ValueError(f"No scene directories found in {input_dir}")
    
    print(f"Found {len(scene_dirs)} scene directories")
    print(f"Class mapping: {class_mapping}\n")
    
    for scene_dir in tqdm(scene_dirs, desc="Processing scenes"):
        output_file = output_path / (scene_dir.name + '.npy')
        try:
            collect_scene_from_annotations(scene_dir, output_file, class_mapping)
        except Exception as e:
            print(f"Error processing {scene_dir.name}: {e}")
    
    print(f"\n✓     Points: {scene_data.shape[0]}")
    print(f"    Unique labels: {np.unique(scene_data[:, 6]).astype(int).tolist()}")
    
    return scene
    return processed_data


def batch_process_directory(input_dir, output_dir, file_pattern='*.txt'):
    """
    Process all files in a directory
    
    Args:
        input_dir: Directory containing your raw point cloud files
        output_dir: Directory to save processed .npy files
        file_pattern: Pattern to match files (e.g., '*.txt', '*.csv', '*.npy')
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    files = list(input_path.glob(file_pattern))
    print(f"Found {len(files)} files to process")
    
    for input_file in tqdm(files, desc="Processing files"):
        output_file = output_path / (input_file.stem + '.npy')
        try:
            prepare_point_cloud_data(str(input_file), str(output_file))
        except Exception as e:
            print(f"Error processing {input_file}: {e}")
    
    print(f"Processing complete! Files saved to: {output_dir}")


def create_sample_data(output_file, num_points=10000, num_classes=6):
    """
    Create sample/dummy data for testing
    Useful for testing the pipeline before you have real data
    """
    # Random 3D coordinates
    xyz = np.random.rand(num_points, 3) * 10  # 10x10x10 space
    
    # Random colors
    rgb = np.random.randint(0, 256, (num_points, 3))
    
    # Random labels (simulate some structure)
    labels = np.random.randint(0, num_classes, num_points)
    
    # Combinecenes', 'sample'], 
                       default='sample', help='Processing mode')
    parser.add_argument('--input', type=str, help='Input directory containing scene folders')
    parser.add_argument('--output', type=str, help='Output directory for .npy files')
    parser.add_argument('--config', type=str, default='twinner01_classes_config.py',
                       help='Path to class config file')
    parser.add_argument('--num_points', type=int, default=10000,
                       help='Number of points for sample mode')
    parser.add_argument('--num_classes', type=int, default=6,
                       help='Number of classes for sample mode')
    
    args = parser.parse_args()
    
    if args.mode == 'scenes':
        if not args.input or not args.output:
            print("Error: --input and --output required for scenes mode")
            print("\nExample:")
            print("  python twinner01_prepare_data.py --mode scenes \\")
            print("    --input raw_data \\")
            print("    --output data/twinner01_custom")
            exit(1)
        
        # Load class mapping from config
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(args.config)))
        from twinner01_classes_config import class2label
        
        batch_process_scenes(args.input, args.output, class2label
                       help='Number of classes for sample mode')
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        if not args.input or not args.output:
            print("Error: --input and --output required for single mode")
            exit(1)
        prepare_point_cloud_data(args.input, args.output)
    
    elif args.mode == 'batch':
        if not args.input or not args.output:
            print("Error: --input and --output required for batch mode")
            exit(1)
        batch_process_directory(args.input, args.output, args.pattern)
    
    elif args.mode == 'sample':
        output = args.output if args.output else 'data/twinner01_custom/sample_scene_01.npy'
        os.makedirs(os.path.dirname(output), exist_ok=True)
        create_sample_data(output, args.num_points, args.num_classes)
