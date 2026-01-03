"""
Twinner01 Custom Classes Configuration for Semantic Segmentation
Modify this file to define your own object classes
"""

# Define your custom classes for twinner01 project
# Replace these with your actual object types
TWINNER01_CLASSES = [
    'background',      # Class 0 - usually background/unknown
    'object_type_1',   # Class 1 - e.g., 'car'
    'object_type_2',   # Class 2 - e.g., 'building'
    'object_type_3',   # Class 3 - e.g., 'tree'
    'object_type_4',   # Class 4 - e.g., 'road'
    'object_type_5',   # Class 5 - e.g., 'pedestrian'
    # Add more classes as needed
]

# Create mapping dictionaries
class2label = {cls: i for i, cls in enumerate(TWINNER01_CLASSES)}
label2class = {i: cls for i, cls in enumerate(TWINNER01_CLASSES)}

# Number of classes
NUM_CLASSES = len(TWINNER01_CLASSES)

# Optional: Define colors for visualization (RGB)
CLASS_COLORS = {
    'background': [128, 128, 128],      # Gray
    'object_type_1': [255, 0, 0],       # Red
    'object_type_2': [0, 255, 0],       # Green
    'object_type_3': [0, 0, 255],       # Blue
    'object_type_4': [255, 255, 0],     # Yellow
    'object_type_5': [255, 0, 255],     # Magenta
}

# Print configuration
def print_config():
    print(f"Twinner01 Project Configuration")
    print(f"Number of classes: {NUM_CLASSES}")
    print(f"Classes: {TWINNER01_CLASSES}")
    print(f"Class to label mapping: {class2label}")

if __name__ == '__main__':
    print_config()
