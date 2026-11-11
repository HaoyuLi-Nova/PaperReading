from typing import Dict, List

import cv2
import numpy as np
import torch
from torchvision.transforms import Normalize
from ..constants import IMAGE_SIZE, IMAGE_MEAN, IMAGE_STD
from .utils import (convert_cvimg_to_tensor,
                    expand_to_aspect_ratio,
                    generate_image_patch_cv2)

DEFAULT_MEAN = 255. * np.array([0.485, 0.456, 0.406])
DEFAULT_STD = 255. * np.array([0.229, 0.224, 0.225])

    
class Dataset(torch.utils.data.Dataset):

    def __init__(self,
                 img_cv2: np.array,
                 bbox_center: np.array,
                 bbox_scale: np.array,
                 cam_int: np.array,
                 train: bool = False,
                 img_path = None,
                 **kwargs):
        super().__init__()
        self.img_cv2 = img_cv2
        self.img_path = img_path
        # self.boxes = boxes

        assert train == False, "ViTDetDataset is only for inference"
        self.train = train
        self.img_size = IMAGE_SIZE
        self.cam_int = cam_int
        self.mean = 255. * np.array(IMAGE_MEAN)
        self.std = 255. * np.array(IMAGE_STD)
        self.normalize_img = Normalize(mean=IMAGE_MEAN,
                                    std=IMAGE_STD)
        self.center = bbox_center
        self.scale = bbox_scale
        self.personid = np.arange(len(self.center), dtype=np.int32)


    def __len__(self) -> int:
        return len(self.personid)

    def __getitem__(self, idx: int):

        center = self.center[idx]
        center_x = center[0]
        center_y = center[1]

        scale = self.scale[idx]
        BBOX_SHAPE = None
        bbox_size = expand_to_aspect_ratio(scale*200, target_aspect_ratio=BBOX_SHAPE).max()

        patch_width = patch_height = self.img_size
        cvimg = self.img_cv2

        img_patch_cv, trans = generate_image_patch_cv2(cvimg,
                                                    center_x, center_y,
                                                    bbox_size, bbox_size,
                                                    patch_width, patch_height,
                                                    False, 1.0, 0,
                                                    border_mode=cv2.BORDER_CONSTANT)


        img_patch = convert_cvimg_to_tensor( img_patch_cv[:, :, ::-1])
        # apply normalization
        # img_patch = self.normalize_img(torch.tensor(img_patch))
        for n_c in range(min(self.img_cv2.shape[2], 3)):
            img_patch[n_c, :, :] = (img_patch[n_c, :, :] - self.mean[n_c]) / self.std[n_c]

        item = {
            'img': img_patch,
            'personid': int(self.personid[idx]),
        }
        item['imgname'] = str(self.img_path)
        item['box_center'] = self.center[idx]
        item['box_size'] =  bbox_size
        item['img_size'] = 1.0 * np.array([cvimg.shape[0], cvimg.shape[1]])
        item['cam_int'] = self.cam_int
        return item


class Dataset_video(torch.utils.data.Dataset):
    def __init__(self,
                 img_cv2_list: List[np.array], 
                 bbox_center_list: List[np.array],
                 bbox_scale_list: List[np.array],
                 cam_int_list: List[np.array],
                 train: bool = False,
                 **kwargs):
        super().__init__()
        self.img_cv2_list = img_cv2_list
        
        assert train == False, "Dataset_video is only for inference"
        self.train = train
        self.img_size = IMAGE_SIZE
        
        # Initialize lists to hold flattened data
        self.center = []
        self.scale = []
        self.personid = []
        self.image_indices = []
        self.cam_int = []

        # Process annotations for each image
        person_counter = 0
        for img_idx, (centers, scales) in enumerate(zip(bbox_center_list, bbox_scale_list)):
            centers = centers.astype(np.float32)
            scales = scales.astype(np.float32)
            personids = np.arange(len(centers), dtype=np.int32) + person_counter
            person_counter += len(centers)

            self.center.append(centers)
            self.scale.append(scales)
            self.personid.append(personids)
            self.image_indices.extend([img_idx] * len(centers))
            self.cam_int.extend([cam_int_list[img_idx]] * len(centers))

        # Flatten the lists
        self.center = np.concatenate(self.center, axis=0)
        self.scale = np.concatenate(self.scale, axis=0)
        self.personid = np.concatenate(self.personid, axis=0)
        self.image_indices = np.array(self.image_indices)
        self.cam_int = np.array(self.cam_int)

        # Setup normalization
        self.mean = 255. * np.array(IMAGE_MEAN)
        self.std = 255. * np.array(IMAGE_STD)
        self.normalize_img = Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)

    def __len__(self) -> int:
        return len(self.personid)

    def __getitem__(self, idx: int):
        img_idx = self.image_indices[idx]
        cvimg = self.img_cv2_list[img_idx]

        center = self.center[idx]
        center_x = center[0]
        center_y = center[1]

        scale = self.scale[idx]
        BBOX_SHAPE = None
        bbox_size = expand_to_aspect_ratio(scale*200, target_aspect_ratio=BBOX_SHAPE).max()

        patch_width = patch_height = self.img_size

        img_patch_cv, trans = generate_image_patch_cv2(cvimg,
                                                    center_x, center_y,
                                                    bbox_size, bbox_size,
                                                    patch_width, patch_height,
                                                    False, 1.0, 0,
                                                    border_mode=cv2.BORDER_CONSTANT)

        img_patch = convert_cvimg_to_tensor(img_patch_cv[:, :, ::-1])
        
        # Apply normalization
        for n_c in range(min(cvimg.shape[2], 3)):
            img_patch[n_c, :, :] = (img_patch[n_c, :, :] - self.mean[n_c]) / self.std[n_c]

        item = {
            'img': img_patch,
            'personid': int(self.personid[idx]),
            'image_idx': int(img_idx)
        }
        item['box_center'] = self.center[idx]
        item['box_size'] = bbox_size
        item['img_size'] = 1.0 * np.array([cvimg.shape[0], cvimg.shape[1]])
        item['cam_int'] = self.cam_int[idx]
        return item
