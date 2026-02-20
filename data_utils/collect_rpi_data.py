import os
import sys
import numpy as np
import open3d as o3d
import glob
import colorsys

ROOT = "data/rpi_custom_dataset_01"

def generate_color_map(class_list):
    color_map = {}
    n = len(class_list)
    
    for i, name in enumerate(class_list):
        hue = i / n
        rgb = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
        color_map[name] = tuple(int(c) for c in rgb)
        
    return color_map

def collect_points_labels(anno_dir):
    points_list = []
    for f in glob.glob(os.path.join(anno_dir, "*.txt")):
        cls = os.path.basename(f).rsplit("_", 1)[0]
        print(f)
        if cls not in classes:
            cls = 'clutter'
        points = np.loadtxt(f, delimiter=",")
        if points.ndim == 1:
            points = points.reshape(1, -1)
        labels = np.ones((points.shape[0],1)) * class2labels[cls]
        points_list.append(np.concatenate([points, labels], 1))
    data = np.concatenate(points_list, 0)
    xyz_min = np.amin(data, axis=0)[0:3]
    data[:, 0:3] -= xyz_min
    return data

if __name__ == "__main__":
    anno_dir = os.path.join(ROOT,"scene_1", "Annotations")
    anno = os.listdir(anno_dir)
    print(f"Number of Annotations: {len(anno)}")

    classes = list(set(map(lambda x: x.rsplit("_", 1)[0], anno)))
    class2labels = {cls: i for i, cls in enumerate(classes)}
    label2color = generate_color_map(classes)
    colors = list( label2color.items() )

    print(f"Num Classes: {len(classes)}")

    data = collect_points_labels(anno_dir)

    out_file = os.path.join(ROOT, "scene_1.npy")

    np.save(out_file, data)
    print(f"Saved as: {out_file}")
    
    




