#!/usr/bin/env python3
import os
import sys
import shutil
import argparse
import subprocess
from pathlib import Path

try:
    from PIL import Image
    import matplotlib.pyplot as plt
except ImportError:
    print("Please install dependencies: pip install pillow matplotlib")
    sys.exit(1)


class ImageViewer:
    """Image viewer with keyboard interaction"""
    
    def __init__(self):
        self.user_choice = None
        self.fig = None
    
    def on_key(self, event):
        """Keyboard event handler"""
        if event.key in ['y', 'Y']:
            self.user_choice = 'y'
            plt.close(self.fig)
        elif event.key in ['n', 'N']:
            self.user_choice = 'n'
            plt.close(self.fig)
        elif event.key in ['q', 'Q', 'escape']:
            self.user_choice = 'q'
            plt.close(self.fig)
    
    def display_and_wait(self, image_path, relative_path):
        """Display image and wait for user input"""
        try:
            self.user_choice = None
            img = Image.open(image_path)
            
            self.fig = plt.figure(figsize=(12, 9))
            plt.imshow(img)
            plt.axis('off')
            
            # Add title and instructions
            title_text = f"Directory: {relative_path}\n\nPress 'Y' to Select | 'N' to Skip | 'Q' or ESC to Quit"
            plt.title(title_text, fontsize=12, pad=20)
            plt.tight_layout()
            
            # Bind keyboard events
            self.fig.canvas.mpl_connect('key_press_event', self.on_key)
            
            # Show image and wait for user action
            plt.show()
            
            return self.user_choice
            
        except Exception as e:
            print(f"Failed to display image {image_path}: {e}")
            return None


def process_subdirectory(subdir_path, relative_path, source_base, target_base, object_base):
    """
    Process a single subdirectory
    
    Args:
        subdir_path: Full path to subdirectory
        relative_path: Relative path (including category and subdirectory name)
        source_base: Source base directory (data)
        target_base: Target base directory (data_sandwich)
        object_base: Object base directory (data_object)
    
    Returns:
        True if processed successfully, False otherwise
    """
    image_path = os.path.join(subdir_path, "image.jpg")
    
    if not os.path.exists(image_path):
        print(f"Warning: Image not found {image_path}")
        return False
    
    # Display image and get user input
    print(f"\n{'='*60}")
    print(f"Current directory: {relative_path}")
    print(f"Full path: {subdir_path}")
    print(f"{'='*60}")
    print("Displaying image, please press a key in the image window...")
    
    viewer = ImageViewer()
    user_choice = viewer.display_and_wait(image_path, relative_path)
    
    if user_choice is None:
        print("Failed to display image")
        return False
    elif user_choice == 'q':
        print("\nExiting program")
        sys.exit(0)
    elif user_choice == 'n':
        print("Skipping this data")
        return False
    elif user_choice == 'y':
        print("Selected this data, processing...")
    else:
        print("Unknown choice, skipping")
        return False
    
    # Move subdirectory to target location
    target_path = os.path.join(target_base, relative_path)
    target_dir = os.path.dirname(target_path)
    
    try:
        # Create target directory
        os.makedirs(target_dir, exist_ok=True)
        
        # Copy entire subdirectory
        if os.path.exists(target_path):
            print(f"Warning: Target path already exists {target_path}, will overwrite")
            shutil.rmtree(target_path)
        
        print(f"Copying: {subdir_path} -> {target_path}")
        shutil.copytree(subdir_path, target_path)
        
        # Rename obj_mask.png to object_mask.png if it exists
        old_mask_path = os.path.join(target_path, "obj_mask.png")
        new_mask_path = os.path.join(target_path, "object_mask.png")
        if os.path.exists(old_mask_path):
            os.rename(old_mask_path, new_mask_path)
            print(f"Renamed: obj_mask.png -> object_mask.png")
        
        # Copy obj_pcd_h_align.obj from data_object
        object_source_path = os.path.join(object_base, relative_path, "obj_pcd_h_align.obj")
        object_target_path = os.path.join(target_path, "obj_pcd_h_align.obj")
        
        if os.path.exists(object_source_path):
            print(f"Copying: {object_source_path} -> {object_target_path}")
            shutil.copy2(object_source_path, object_target_path)
        else:
            print(f"Warning: Not found {object_source_path}")
        
        # Run camera.py
        print("\nRunning camera.py...")
        camera_script = os.path.join(os.path.dirname(__file__), "..", "prepare", "camera.py")
        camera_cmd = ["python", camera_script, "--data_dir", target_path]
        result = subprocess.run(camera_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"camera.py execution failed:")
            print(result.stderr)
        else:
            print("camera.py executed successfully")
        
        # Run normal.py
        print("\nRunning normal.py...")
        normal_script = os.path.join(os.path.dirname(__file__), "..", "prepare", "normal.py")
        normal_cmd = ["python", normal_script, "--data_dir", target_path]
        result = subprocess.run(normal_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"normal.py execution failed:")
            print(result.stderr)
        else:
            print("normal.py executed successfully")
        
        print(f"\n✓ Successfully processed: {relative_path}")
        return True
        
    except Exception as e:
        print(f"Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def scan_and_process(source_dir, target_dir, object_dir, category=None):
    """
    Scan and process data directories
    
    Args:
        source_dir: Source data directory (data)
        target_dir: Target data directory (data_sandwich)
        object_dir: Object data directory (data_object)
        category: Specified category name, if None process all categories
    """
    source_path = Path(source_dir)
    
    if not source_path.exists():
        print(f"Error: Source directory does not exist: {source_dir}")
        return
    
    # Determine categories to process
    if category:
        categories = [category]
        category_path = source_path / category
        if not category_path.exists():
            print(f"Error: Category directory does not exist: {category_path}")
            return
    else:
        # Get all category directories
        categories = [d.name for d in source_path.iterdir() if d.is_dir()]
        categories.sort()
    
    print(f"Found {len(categories)} categories")
    print(f"Categories: {', '.join(categories)}")
    
    total_processed = 0
    total_selected = 0
    
    # Iterate through each category
    for cat in categories:
        cat_path = source_path / cat
        print(f"\n{'#'*60}")
        print(f"Processing category: {cat}")
        print(f"{'#'*60}")
        
        # Get all subdirectories under this category
        subdirs = [d for d in cat_path.iterdir() if d.is_dir()]
        subdirs.sort()
        
        print(f"Found {len(subdirs)} subdirectories")
        
        for subdir in subdirs:
            total_processed += 1
            relative_path = os.path.join(cat, subdir.name)
            
            if process_subdirectory(str(subdir), relative_path, source_dir, target_dir, object_dir):
                total_selected += 1
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Total processed: {total_processed} directories")
    print(f"Selected: {total_selected} directories")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Data Selector Tool - Browse images and selectively move data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all categories
  python data_selector.py
  
  # Process only sandwich category
  python data_selector.py --category sandwich
  
  # Specify custom paths
  python data_selector.py --source ./data --target ./data_sandwich --object ./data_object
        """
    )
    
    parser.add_argument(
        '--source',
        type=str,
        default='/home/xhsystem/Code/Term7/Ca3OH1/data',
        help='Source data directory (default: /home/xhsystem/Code/Term7/Ca3OH1/data)'
    )
    
    parser.add_argument(
        '--target',
        type=str,
        default='/home/xhsystem/Code/Term7/Ca3OH1/data_sandwich',
        help='Target data directory (default: /home/xhsystem/Code/Term7/Ca3OH1/data_sandwich)'
    )
    
    parser.add_argument(
        '--object',
        type=str,
        default='/home/xhsystem/Code/Term7/Ca3OH1/data_object',
        help='Object data directory (default: /home/xhsystem/Code/Term7/Ca3OH1/data_object)'
    )
    
    parser.add_argument(
        '--category',
        type=str,
        default=None,
        help='Specify category name to process (e.g., bottle), if not specified will process all categories'
    )
    
    args = parser.parse_args()
    
    # Convert to absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    source_dir = os.path.join(script_dir, args.source) if not os.path.isabs(args.source) else args.source
    target_dir = os.path.join(script_dir, args.target) if not os.path.isabs(args.target) else args.target
    object_dir = os.path.join(script_dir, args.object) if not os.path.isabs(args.object) else args.object
    
    print(f"Source directory: {source_dir}")
    print(f"Target directory: {target_dir}")
    print(f"Object directory: {object_dir}")
    
    # Create target directory
    os.makedirs(target_dir, exist_ok=True)
    
    # Start processing
    scan_and_process(source_dir, target_dir, object_dir, args.category)


if __name__ == "__main__":
    main()
