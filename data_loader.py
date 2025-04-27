import os
import random
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image
import numpy as np
import scipy.ndimage
import torch
import cv2


class ImageFolder(data.Dataset):
    def __init__(self, root, image_size=256, mode='train', augmentation_prob=0.4, exp_or_sim='sim', image_type='8bits',config=None):
        self.root = root  # 'xxxxxxx/trian_valid' or 'xxxxxxx/test'
        self.mode = mode
        print(f'\n=============> Loading data from {self.root} for {mode}. ==========>\n')
        self.image_type = image_type

        train_input_list = []
        train_gt_list = []
        valid_input_list = []
        valid_gt_list = []
        test_input_list = []
        test_gt_list = []

        if self.mode == 'train' or self.mode == 'valid':
            folders = config.selected_train_valid_fold  # ['2nanoholes', '3nanoholes', ..., '10nanoholes']
        elif self.mode == 'test':
            folders = config.selected_test_fold

        for fold in folders:
            fold_input = os.path.join(root, fold, 'img')
            fold_gt = os.path.join(root, fold, 'mask')

            # get the name of fold
            fold_name = fold.split('/')[-1]

            num_data_in_fold = len(os.listdir(fold_input))
            upper_limit = 100000  # determine the amount of training dataset

            if self.mode == 'train' or self.mode == 'valid':
                '''
                valid_rate = config.valid_rate  # e.g., 10%
                valid_num = int(valid_rate * num_data_in_fold)

                for i, item in enumerate(sorted(os.listdir(fold_input))):
                    if i < valid_num:
                        valid_input_list.append(os.path.join(fold_input, item))
                        valid_gt_list.append(os.path.join(fold_gt, item))
                    elif i >= valid_num and i < upper_limit:
                        train_input_list.append(os.path.join(fold_input, item))
                        train_gt_list.append(os.path.join(fold_gt, item))
                    else:
                        break
                '''
                '''
                valid_rate = config.valid_rate  # e.g., 10%
                valid_num = int(valid_rate * num_data_in_fold)
                all_items = os.listdir(fold_input)

                # Shuffle the items randomly
                random.shuffle(all_items)

                # Loop through the shuffled items
                for i, item in enumerate(all_items):
                    bmp_path = os.path.join(fold_input, item.replace('.tif', '.bmp'))
                    gt_bmp_path = os.path.join(fold_gt, item.replace('.tif', '.bmp'))

                    if os.path.exists(bmp_path):
                        input_path = bmp_path
                    else:
                        input_path = os.path.join(fold_input, item)

                    if os.path.exists(gt_bmp_path):
                        gt_path = gt_bmp_path
                    else:
                        gt_path = os.path.join(fold_gt, item)  # Fallback to original format

                    # **Check if the ground truth image is completely black**
                    gt_image = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale
                    if gt_image is None:
                        continue  # Skip if the image cannot be loaded
                    if np.all(gt_image == 0):
                        continue  # Skip processing if GT is all black

                    # Assign the first part as validation and the later part as training
                    if i < valid_num:
                        valid_input_list.append(input_path)
                        valid_gt_list.append(gt_path)
                    elif i >= valid_num and i < upper_limit:
                        train_input_list.append(input_path)
                        train_gt_list.append(gt_path)
                    else:
                        break

                '''
                valid_rate = config.valid_rate  # e.g., 10%
                valid_num = int(valid_rate * num_data_in_fold)
                all_items = os.listdir(fold_input)

                # Shuffle the items randomly
                random.shuffle(all_items)

                # Loop through the shuffled items
                for i, item in enumerate(all_items):
                    bmp_path = os.path.join(fold_input, item.replace('.tif', '.bmp'))
                    gt_bmp_path = os.path.join(fold_gt, item.replace('.tif', '.bmp'))

                    if os.path.exists(bmp_path):
                        input_path = bmp_path
                    else:
                        input_path = os.path.join(fold_input, item)  # Fallback to original format

                    if os.path.exists(gt_bmp_path):
                        gt_path = gt_bmp_path
                    else:
                        gt_path = os.path.join(fold_gt, item)  # Fallback to original format

                    # Assign the first part as validation and the later part as training
                    if i < valid_num:
                        valid_input_list.append(input_path)
                        valid_gt_list.append(gt_path)
                    elif i >= valid_num and i < upper_limit:
                        train_input_list.append(input_path)
                        train_gt_list.append(gt_path)
                    else:
                        break
                    


            if self.mode == 'test':
                #for item in range(num_data_in_fold):
                 #   test_input_list.append(os.path.join(fold_input, str(item)+'.tif'))
                 #   test_gt_list.append(os.path.join(fold_gt, str(item)+'.tif'))
                for item in range(num_data_in_fold):
                #for item in range(5000, 7000):  # yuhan

                    tif_path = os.path.join(fold_input, f"{item}.tif")
                    bmp_path = os.path.join(fold_input, f"{item}.bmp")
                    gt_tif_path = os.path.join(fold_gt, f"{item}.tif")
                    gt_bmp_path = os.path.join(fold_gt, f"{item}.bmp")

                    # Check if the file exists and append accordingly
                    if os.path.exists(tif_path):
                        test_input_list.append(tif_path)
                    elif os.path.exists(bmp_path):
                        test_input_list.append(bmp_path)

                    if os.path.exists(gt_tif_path):
                        test_gt_list.append(gt_tif_path)
                    elif os.path.exists(gt_bmp_path):
                        test_gt_list.append(gt_bmp_path)
        if self.mode == 'train':
            self.image_paths = train_input_list
            self.GT_paths = train_gt_list
        elif self.mode == 'valid':
            self.image_paths = valid_input_list
            self.GT_paths = valid_gt_list
        elif self.mode == 'test':
            self.image_paths = test_input_list
            self.GT_paths = test_gt_list

        self.image_size = image_size
        self.config = config
        self.RotationDegree = [0, 90, 180, 270]
        self.augmentation_prob = augmentation_prob
        print("image count in {} path :{}".format(self.mode, len(self.image_paths)))

    def __getitem__(self, index):
        #print(f'index {index}')
        image_path = self.image_paths[index]
        GT_path = self.GT_paths[index]
        image_type = self.image_type

        if image_type == '16bits':
            image = Image.open(image_path).convert("L")
            GT = Image.open(GT_path).convert("L")
            ResizeRange = 256

            # ============================ try float32 ========================== #
            image_array = np.asarray(image).astype(np.float32)
            GT_array   = np.asarray(GT).astype(np.float32)
            # =================================================================== #

            zoom_ratio_x = ResizeRange/image_array.shape[0]
            zoom_ratio_y = ResizeRange/image_array.shape[1]
            image_array_resize = scipy.ndimage.zoom(image_array, zoom=(zoom_ratio_x, zoom_ratio_y), order=1)

            GT_zoom_ratio_x = ResizeRange / GT_array.shape[0]
            GT_zoom_ratio_y = ResizeRange / GT_array.shape[1]
            GT_array_resize = scipy.ndimage.zoom(GT_array, zoom=(GT_zoom_ratio_x, GT_zoom_ratio_y), order=1)

            image_increase = image_array_resize
            GT_increase = GT_array_resize
            if (self.mode == 'train'):

                if random.random() < 0.5:
                    image_increase = np.rot90(image_array_resize)
                    GT_increase = np.rot90(GT_array_resize)

                if random.random() < 0.5:
                    image_increase= np.fliplr(image_increase)
                    GT_increase= np.fliplr(GT_increase)

                if random.random() < 0.5:
                    image_increase = np.flipud(image_increase)
                    GT_increase = np.flipud(GT_increase)

            min_value = np.min(image_increase)
            max_value = np.max(image_increase)

            normalized_image_array = (image_increase - min_value) / (max_value - min_value)
            image_array_nor = (normalized_image_array * 2) - 1
            image_trans = torch.tensor(image_array_nor.copy()).unsqueeze(0)

            GT_trans = torch.tensor(GT_increase.copy()).unsqueeze(0)
            GT_trans = GT_trans/255


        if image_type =='8bits':
            image = Image.open(image_path).convert("L")
            GT    = Image.open(GT_path).convert("L")
            aspect_ratio = image.size[1] / image.size[0]
            Transform = []
            ResizeRange = 256
            t_resize = T.Resize((int(ResizeRange * aspect_ratio), ResizeRange))
            image = t_resize(image)
            GT = t_resize(GT)

            # ==========================================================================#

            if (self.mode == 'train'):
                if self.config.rotate:
                    RotationDegree = random.randint(0, 3)
                    RotationDegree = self.RotationDegree[RotationDegree]
                    if (RotationDegree == 90) or (RotationDegree == 270):
                        aspect_ratio = 1 / aspect_ratio

                    Transform.append(T.RandomRotation((RotationDegree, RotationDegree)))
            # ==========================================================================#

                if random.random() < 0.5:
                    image = F.hflip(image)
                    GT = F.hflip(GT)

                if random.random() < 0.5:
                    image = F.vflip(image)
                    GT = F.vflip(GT)

            if self.config.center_crop:
                CropRange = random.randint(256, 256)
                crop_operation = T.CenterCrop((int(CropRange * aspect_ratio), CropRange))
                image = crop_operation(image)

            Transform.append(T.Resize((256, 256)))
            Transform.append(T.ToTensor())
            Transform = T.Compose(Transform)
            image_trans = Transform(image)
            GT_trans = Transform(GT)

            Norm_ = T.Normalize(mean=0.5, std=0.5)
            image_trans = Norm_(image_trans)

        return image_trans, GT_trans, image_path, GT_path


    def __len__(self):
        """Returns the total number of font files."""
        return len(self.image_paths)


def get_loader(image_path, image_type, image_size, batch_size, num_workers=2, mode='train', augmentation_prob=0.4, exp_or_sim=None,
               config=None):
    """Builds and returns Dataloader."""

    dataset = ImageFolder(root=image_path, image_type=image_type,image_size=image_size, mode=mode, augmentation_prob=augmentation_prob,
                          exp_or_sim=exp_or_sim, config=config)
    if mode == 'train':
        data_loader = data.DataLoader(dataset=dataset,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=num_workers)
    elif mode == 'test':
        data_loader = data.DataLoader(dataset=dataset,
                                      batch_size=1,
                                      shuffle=False,
                                      num_workers=num_workers)
    elif mode == 'valid':
        data_loader = data.DataLoader(dataset=dataset,
                                      batch_size=batch_size,
                                      shuffle=False,
                                      num_workers=num_workers)

    return data_loader
