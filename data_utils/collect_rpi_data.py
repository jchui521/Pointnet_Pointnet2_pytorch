import os
import numpy as np
import glob
from tqdm import tqdm

if __name__ == "__main__":
    # ROOT = "/home/nvidia/Pointnet_Pointnet2_pytorch/rpi_data_raw"
    ROOT = "rpi_data_raw"
    if not os.path.exists(ROOT):
        os.mkdir(ROOT)

    # output_dir = "/home/nvidia/Pointnet_Pointnet2_pytorch/rpi_data"
    output_dir = "rpi_data"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    class_file = "classes.txt"
    with open(class_file, "r") as file:
        classes = [line.rstrip() for line in file.readlines()]
    class2labels = {cls: i for i, cls in enumerate(classes)}

    for scene_dir in os.listdir(ROOT):
        print(f"Processing Scene: {scene_dir}")
        anno_dir = os.path.join(ROOT,scene_dir, "_PointOut")
        anno = os.listdir(anno_dir)
        print(f"Number of Annotations: {len(anno)}")

        points_list = []
        for f in tqdm( glob.glob(os.path.join(anno_dir, "*.xyz")) ):
            cls = os.path.basename(f).rsplit("_", 1)[0]
            if cls not in classes:
                cls = 'Clutter'
            points = np.loadtxt(f, delimiter=",")
            if points.ndim == 1:
                points = points.reshape(1, -1)
            labels = np.ones((points.shape[0],1)) * class2labels[cls]
            points_list.append(np.concatenate([points, labels], 1))
        data = np.concatenate(points_list, 0)
        xyz_min = np.amin(data, axis=0)[0:3]
        data[:, 0:3] -= xyz_min
        
        out_file = os.path.join(output_dir, scene_dir+".npy")

        np.save(out_file, data)
        print(f"Saved as: {out_file}")
        
        




