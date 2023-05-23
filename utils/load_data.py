import sys
import warnings
import numpy as np
from tqdm import tqdm
from PIL import Image
from cv2 import imread
from os import listdir
from os.path import isfile, join
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import is_image_file

from utils import *

cfg = get_cfg()


class pair_path():
    def __init__(self, a_path, b_path):
        self.a_path = a_path
        self.b_path = b_path


class load_focus_seperate(Dataset):
    def __init__(self, dir_path):
        super(load_focus_seperate, self).__init__()
        self.a_path = dir_path
        self.image_path_list = [[] for _ in range(20)]
        self.render_dataset()

    def __getitem__(self, idx):
        return self.image_path_list[idx]

    def __len__(self):
        return len(self.image_path_list)

    def render_dataset(self):
        warnings.filterwarnings("ignore", category=Warning)
        region_dir_list = sorted(
            [f for f in listdir(self.a_path) if not isfile(join(self.a_path, f))])
        for region_dir in region_dir_list:
            region_path = join(self.a_path, region_dir)
            focus_dir_list = sorted(
                [f for f in listdir(region_path) if not isfile(join(region_path, f)) and f[9:12] != "RGB"])
            # paths for one focus group of a images
            for j, focus_dir in enumerate(focus_dir_list):
                focus_path = join(region_path, focus_dir)
                # paths for one group of a images
                img_list = sorted(
                    [f for f in listdir(focus_path) if is_image_file(join(focus_path, f)) and f[9:12] != "RGB"])
                a_image_path = []
                for img_name in img_list:
                    img_path = join(focus_path, img_name)
                    a_image_path.append(img_path)
                self.image_path_list[j].append(a_image_path)


class load_simple(Dataset):
    def __init__(self, dir_path):
        super(load_simple, self).__init__()
        self.a_path = dir_path
        self.image_path_list = []
        self.render_dataset()

    def __getitem__(self, idx):
        return self.image_path_list[idx]

    def __len__(self):
        return len(self.image_path_list)

    def render_dataset(self):
        warnings.filterwarnings("ignore", category=Warning)
        region_dir_list = sorted(
            [f for f in listdir(self.a_path) if not isfile(join(self.a_path, f))])
        for region_dir in region_dir_list:
            region_path = join(self.a_path, region_dir)
            a_img_list = sorted(
                [f for f in listdir(region_path) if is_image_file(join(region_path, f)) and f[9:12] != "RGB"])
            a_image_path = []
            for img_name in a_img_list:
                img_path = join(region_path, img_name)
                a_image_path.append(img_path)
            self.image_path_list.append(a_image_path)


class MS_dataset(Dataset):
    def __init__(self, dir_path, mode):
        super(MS_dataset, self).__init__()
        self.a_path = dir_path + '/a'
        self.b_path = dir_path + '/b'
        self.image_path_list = []
        self.render_dataset(mode)

    def __getitem__(self, idx):
        return self.image_path_list[idx]

    def __len__(self):
        return len(self.image_path_list)

    def render_dataset(self, mode="all"):
        warnings.filterwarnings("ignore", category=Warning)
        region_dir_list = sorted(
            [f for f in listdir(self.a_path) if not isfile(join(self.a_path, f))])
        for region_dir in region_dir_list:
            region_path = join(self.a_path, region_dir)
            focus_dir_list = sorted([f for f in listdir(region_path) if not isfile(join(region_path, f))])

            if mode == "all" or mode == "only_b" or mode == "only_b_cyc":
                # paths for b images in the region
                b_region_path = join(self.b_path, region_dir)
                b_img_list = sorted(
                    [f for f in listdir(b_region_path) if is_image_file(join(region_path, f)) and f[9:12] != "RGB"])
                b_image_path = []
                for img_name in b_img_list:
                    img_path = join(b_region_path, img_name)
                    b_image_path.append(img_path)

            if mode == "only_b_cyc":
                self.image_path_list.append(b_image_path)
                continue

            # paths for one focus group of a images
            for focus_dir in focus_dir_list:
                focus_path = join(region_path, focus_dir)

                # paths for one group of a images
                img_list = sorted(
                    [f for f in listdir(focus_path) if is_image_file(join(focus_path, f)) and f[9:12] != "RGB"])
                a_image_path = []
                for img_name in img_list:
                    img_path = join(focus_path, img_name)
                    a_image_path.append(img_path)

                if mode == "all":
                    # make a pair
                    pair_p = a_image_path
                    pair_p.extend(b_image_path)
                    self.image_path_list.append(pair_p)
                elif mode == "only_a":
                    self.image_path_list.append(a_image_path)
                elif mode == "only_b":
                    self.image_path_list.append(b_image_path)

