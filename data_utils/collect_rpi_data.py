import os
import numpy as np
import glob
import colorsys
import zipfile

def collect_points_labels(anno_dir):
    points_list = []
    for f in glob.glob(os.path.join(anno_dir, "*.xyz")):
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
    ROOT = "rpi_data_raw"

    zip_file_dir = "zips"
    zip_files = os.listdir(zip_file_dir)
    for zip_file in zip_files:
        print(f"Unzipping: {zip_file}")
        zip_file_path = os.path.join(zip_file_dir, zip_file)
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(ROOT)
    classes = []
    scenes = os.listdir(ROOT)
    for scene in scenes:
        anno_dir = os.path.join(ROOT,scene, "_PointOut")
        anno = os.listdir(anno_dir)
        for a in anno:
            if ".xyz" in a:
                cls = a.rsplit("_", 1)[0]
                if cls not in classes:
                    classes.append(cls)
    n = len(classes)
    class2labels = {cls: i for i, cls in enumerate(classes)}
    class_colors = []
    for i, name in enumerate(classes):
        hue = i / n
        rgb = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
        class_colors.append([c for c in rgb])
    class_colors = np.array(class_colors)

    for scene_dir in os.listdir(ROOT):
        print(f"Processing Scene: {scene_dir}")
        anno_dir = os.path.join(ROOT,scene_dir, "_PointOut")
        anno = os.listdir(anno_dir)
        print(f"Number of Annotations: {len(anno)}")

        data = collect_points_labels(anno_dir)

        out_file = os.path.join(ROOT, "scene_1.npy")

        np.save(out_file, data)
        print(f"Saved as: {out_file}")
        
        




