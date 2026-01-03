"""
Twinner01 Custom Classes Configuration (template copy)
Customize this file in-place for new Twinner variants (e.g., oh1, oh2).
"""

TWINNER01_CLASSES = [
    'background',      # Class 0 - usually background/unknown
    'object_type_1',   # Class 1 - e.g., 'car'
    'object_type_2',   # Class 2 - e.g., 'building'
    'object_type_3',   # Class 3 - e.g., 'tree'
    'object_type_4',   # Class 4 - e.g., 'road'
    'object_type_5',   # Class 5 - e.g., 'pedestrian'
]

class2label = {cls: i for i, cls in enumerate(TWINNER01_CLASSES)}
label2class = {i: cls for i, cls in enumerate(TWINNER01_CLASSES)}
NUM_CLASSES = len(TWINNER01_CLASSES)

CLASS_COLORS = {
    'background': [128, 128, 128],
    'object_type_1': [255, 0, 0],
    'object_type_2': [0, 255, 0],
    'object_type_3': [0, 0, 255],
    'object_type_4': [255, 255, 0],
    'object_type_5': [255, 0, 255],
}

def print_config():
    print("Twinner01 Template Configuration")
    print(f"Number of classes: {NUM_CLASSES}")
    print(f"Classes: {TWINNER01_CLASSES}")
    print(f"Class to label mapping: {class2label}")

if __name__ == '__main__':
    print_config()
