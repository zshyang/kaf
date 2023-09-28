import random
from random import choice

import cv2
import numpy as np
import torch
from batchgenerators.transforms.abstract_transforms import (Compose,
                                                            RndTransform)
from batchgenerators.transforms.crop_and_pad_transforms import \
    RandomCropTransform
from batchgenerators.transforms.spatial_transforms import (MirrorTransform,
                                                           SpatialTransform)
from batchgenerators.utilities.file_and_folder_operations import *
from torch.utils.data.dataset import Dataset

from .match_utils import (gather_keypoints, gather_match_information,
                          label_to_keypoints, prepare_transform)
from .utils import *


def visualize_keypoints(image, keypoints, save_path):
    cv2_keypoints = [
        cv2.KeyPoint(x=float(pt[1]), y=float(pt[0]), size=1) for pt in keypoints
    ]

    cv2_image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255
    cv2_image = cv2_image.astype(np.uint8)

    image_keypoints = cv2.drawKeypoints(
        cv2_image, cv2_keypoints, None,
        color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    cv2.imwrite(save_path, image_keypoints)


def augment_image_and_label(image, label, transform):
    # the image and label should be [batch, c, x, y, z], this is the adapatation for using batchgenerators :)
    data_dict = {'data': image[None, None], 'seg': label[None]}

    data_dict = transform(**data_dict)

    augmented_image = data_dict.get('data')[0]
    augmented_label = data_dict.get('seg')[0]

    return augmented_image, augmented_label


class ACDC(Dataset):

    def __init__(self, keys, purpose, args):
        self.data_dir = args.data_dir
        self.patch_size = args.patch_size
        self.purpose = purpose
        self.classes = args.classes
        self.do_contrast = args.do_contrast
        self.files = []
        if self.do_contrast:
            # we do not pre-load all data, instead, load data in the get item function
            self.slice_position = []
            self.partition = []
            self.slices = []
            self.total_slices = []
            for key in keys:
                frames = subfiles(
                    join(
                        self.data_dir, 'patient_%03d' % key
                    ), False, None, ".npy", True
                )
                for frame in frames:
                    image = np.load(
                        join(self.data_dir, 'patient_%03d' % key, frame)
                    )[0]
                    for i in range(image.shape[0]):
                        self.files.append(
                            join(self.data_dir, 'patient_%03d' % key, frame)
                        )
                        self.slices.append(i)
                        self.slice_position.append(float(i+1)/image.shape[0])
                        part = image.shape[0] / 4.0
                        if part - int(part) >= 0.5:
                            part = int(part + 1)
                        else:
                            part = int(part)
                        self.partition.append(max(0, min(int(i//part), 3)+1))
                        self.total_slices.append(image.shape[0])
        else:
            for key in keys:
                frames = subfiles(
                    join(self.data_dir, 'patient_%03d' % key), False, None, ".npy", True)
                for frame in frames:
                    image = np.load(
                        join(self.data_dir, 'patient_%03d' % key, frame))
                    for i in range(image.shape[1]):
                        self.files.append(image[:, i])
        print(f'dataset length: {len(self.files)}')

    def _get_neighbor_frame_path(self, index):
        # find a neighbor image
        current_file_path = self.files[index]
        current_frame_index = int(
            current_file_path.split('/')[-1].split('.')[0][5:]
        )

        neighbor_frame_index = int(
            0.18 * random.uniform(-1, 1) * self.total_frames[index]) + current_frame_index
        while neighbor_frame_index == current_frame_index or neighbor_frame_index < 0 or neighbor_frame_index >= self.total_frames[index]:
            neighbor_frame_index = int(
                0.18 * random.uniform(-1, 1) * self.total_frames[index]) + current_frame_index
        if current_file_path.endswith('.npy'):
            npy_file_name = 'frame' + f'{neighbor_frame_index:03d}' + '.npy'
        else:
            npy_file_name = 'frame' + f'{neighbor_frame_index:03d}' + '.npz'
        # npy_file_name = 'frame' + f'{neighbor_frame_index:03d}' + '.npy'
        neighbor_file_path = current_file_path.split(
            '/')[:-1] + [npy_file_name]
        neighbor_file_path = '/'.join(neighbor_file_path)

        return neighbor_file_path

    def _get_neighbor_slice_index(self, index):
        current_slice = self.slices[index]
        current_total_slices = self.total_slices[index]
        threshold = 0.18 if 2. / current_total_slices > 0.18 else 2. / current_total_slices
        random_number = threshold * random.uniform(-1, 1)
        neighbor_slice_index = random_number * current_total_slices + current_slice
        neighbor_slice_index = int(neighbor_slice_index)
        while (neighbor_slice_index == current_slice) or (neighbor_slice_index < 0) or (neighbor_slice_index >= current_total_slices):
            random_number = threshold * random.uniform(-1, 1)
            neighbor_slice_index = random_number * current_total_slices + current_slice
            neighbor_slice_index = int(neighbor_slice_index)
        return neighbor_slice_index

    def _sup_process(self, image, label):

        debug = False

        keypoints, _ = gather_keypoints(image)

        if keypoints is None:
            return None, None, None

        # convert keypoints to label array
        label_array = np.ones_like(image) * (-1)
        label_array[keypoints[:, 0], keypoints[:, 1]] = np.arange(
            keypoints.shape[0]
        )

        if debug:
            import matplotlib.pyplot as plt
            plt.imshow(label)
            plt.savefig('save_sup_label_before.png')

        # resize image
        if debug:
            self.purpose = 'test'

        if self.purpose == 'train':
            image, coord = pad_and_or_crop(
                image, self.patch_size, mode='random'
            )
            label_array, _ = pad_and_or_crop(
                label_array, self.patch_size, mode='fixed', coords=coord
            )
            label, _ = pad_and_or_crop(
                label, self.patch_size, mode='fixed', coords=coord
            )
            concat_label = np.concatenate(
                (label_array[None], label[None]), axis=0
            )
            transform = prepare_transform(self.patch_size)
            image, concat_label = augment_image_and_label(
                image, concat_label, transform
            )
            label_array, label = concat_label[0], concat_label[1]
            image = image[0]

        else:
            image, coord = pad_and_or_crop(
                image, self.patch_size, mode='centre'
            )
            label_array, _ = pad_and_or_crop(
                label_array, self.patch_size, mode='fixed', coords=coord
            )
            label, _ = pad_and_or_crop(
                label, self.patch_size, mode='fixed', coords=coord
            )
        # convert the label back to keypoints
        keypoints1, keypoints1_mask = label_to_keypoints(
            label_array[None], num_keypoints=keypoints.shape[0]
        )
        keypoints1_mask = (keypoints1_mask == 1)

        if debug:

            visualize_keypoints(image, keypoints1, 'save_sup_kp.png')

            import matplotlib.pyplot as plt
            plt.imshow(label)
            plt.savefig('save_sup_label_after.png')

        return image[None], label[None], keypoints1

    def __getitem__(self, index):
        if self.do_contrast:
            volume = np.load(self.files[index]).astype(np.float32)[0]
            img = volume[self.slices[index]]

            neighbor_slice_index = self._get_neighbor_slice_index(index)
            neighbor_image = volume[neighbor_slice_index]

            img1, nimg1, reordered_kp1, reordered_nkp1, match, nmatch = gather_match_information(
                img, neighbor_image, self.patch_size
            )

            if img1 is None:
                return None

            return img1, nimg1, reordered_kp1, reordered_nkp1, match, nmatch, self.slice_position[index]

        else:
            label = self.files[index][1]

            while label.max() > 3 or label.min() < 0:
                print(f'index {index} is not valid, resample')
                index = random.randint(0, len(self.files) - 1)
                label = self.files[index][1]
            img = self.files[index][0].astype(np.float32)
            label = self.files[index][1]
            # img, label = self.prepare_supervised(img, label)
            img, label, keypoints = self._sup_process(img, label)
            if img is None or label is None or keypoints is None:
                return None
            return img, label, keypoints, keypoints[:, 0], img, keypoints

    # this function for normal supervised training

    def prepare_supervised(self, img, label):
        if self.purpose == 'train':
            # resize image
            img, coord = pad_and_or_crop(img, self.patch_size, mode='random')
            label, _ = pad_and_or_crop(
                label, self.patch_size, mode='fixed', coords=coord)
            # the image and label should be [batch, c, x, y, z], this is the adapatation for using batchgenerators :)
            data_dict = {'data': img[None, None], 'seg': label[None, None]}
            tr_transforms = []
            tr_transforms.append(MirrorTransform((0, 1)))
            tr_transforms.append(RndTransform(SpatialTransform(self.patch_size, list(np.array(self.patch_size)//2),
                                                               True, (100.,
                                                                      350.), (14., 17.),
                                                               True, (0, 2.*np.pi), (-0.000001,
                                                                                     0.00001), (-0.000001, 0.00001),
                                                               True, (
                                                                   0.7, 1.3), 'constant', 0, 3, 'constant', 0, 0,
                                                               random_crop=False), prob=0.67, alternative_transform=RandomCropTransform(self.patch_size)))

            train_transform = Compose(tr_transforms)
            data_dict = train_transform(**data_dict)
            img = data_dict.get('data')[0]
            label = data_dict.get('seg')[0]
            return img, label
        else:
            # resize image
            img, coord = pad_and_or_crop(img, self.patch_size, mode='centre')
            label, _ = pad_and_or_crop(
                label, self.patch_size, mode='fixed', coords=coord)
            return img[None], label[None]

    # use this function for contrastive learning
    def prepare_contrast(self, img):
        # resize image
        img, coord = pad_and_or_crop(img, self.patch_size, mode='random')
        # the image and label should be [batch, c, x, y, z], this is the adapatation for using batchgenerators :)
        data_dict = {'data': img[None, None]}
        tr_transforms = []
        tr_transforms.append(MirrorTransform((0, 1)))
        tr_transforms.append(RndTransform(SpatialTransform(self.patch_size, list(np.array(self.patch_size)//2),
                                                           True, (100.,
                                                                  350.), (14., 17.),
                                                           True, (0, 2.*np.pi), (-0.000001,
                                                                                 0.00001), (-0.000001, 0.00001),
                                                           True, (
                                                               0.7, 1.3), 'constant', 0, 3, 'constant', 0, 0,
                                                           random_crop=False), prob=0.67, alternative_transform=RandomCropTransform(self.patch_size)))

        train_transform = Compose(tr_transforms)
        data_dict1 = train_transform(**data_dict)
        img1 = data_dict1.get('data')[0]
        data_dict2 = train_transform(**data_dict)
        img2 = data_dict2.get('data')[0]
        return img1, img2

    def __len__(self):
        return len(self.files)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument("--data_dir", type=str, default="d:/data/acdc/acdc_contrastive/contrastive/2d/")
    parser.add_argument("--data_dir", type=str,
                        default="/afs/crc.nd.edu/user/d/dzeng2/data/acdc/acdc_contrastive/contrastive/2d/")
    parser.add_argument("--patch_size", type=tuple, default=(352, 352))
    parser.add_argument("--classes", type=int, default=4)
    parser.add_argument("--do_contrast", default=True, action='store_true')
    parser.add_argument("--slice_threshold", type=float, default=0.5)
    args = parser.parse_args()

    train_dataset = ACDC(keys=list(range(1, 101)), purpose='train', args=args)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=32,
                                                   shuffle=True,
                                                   num_workers=8,
                                                   drop_last=False)

    pp = []
    for batch_idx, tup in enumerate(train_dataloader):
        print(f'the {batch_idx}th/{len(train_dataloader)} minibatch...')
        img1, img2, slice_position, partition = tup
        batch_size = img1.shape[0]
        # print(f'batch_size:{batch_size}, slice_position:{slice_position}')
        slice_position = slice_position.contiguous().view(-1, 1)
        mask = (torch.abs(slice_position.T.repeat(batch_size, 1) -
                slice_position.repeat(1, batch_size)) < args.slice_threshold).float()
        # count how many positive pair in each batch
        for i in range(batch_size):
            pp.append(2*mask[i].sum()-1)
    pp = np.asarray(pp)
    pp_mean = np.mean(pp)
    pp_std = np.std(pp)
    print(f'average number of positive pairs mean:{pp_mean}, std:{pp_std}')
