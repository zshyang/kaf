import os
import pickle
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

from .utils import *


def gather_keypoints(image):
    # normalize the image
    image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255
    image = image.astype(np.uint8)
    sift = cv2.SIFT_create()
    image = np.squeeze(image, 0)
    keypoints, features = sift.detectAndCompute(image, None)
    # convert keypoints to numpy array
    # the order is y, x
    keypoints = np.array([[kp.pt[1], kp.pt[0]] for kp in keypoints])
    keypoints = np.round(keypoints).astype(np.int32)

    if keypoints.shape[0] == 0:
        return None, None

    # random permute the keypoints and features
    perm = np.random.permutation(len(keypoints))
    keypoints = keypoints[perm]
    features = features[perm]
    return keypoints, features


def manhattan_distance(keypoints1, keypoints2):
    # Expand the keypoints matrices to have a third axis
    keypoints1 = keypoints1[:, np.newaxis, :]
    keypoints2 = keypoints2[np.newaxis, :, :]
    # Compute the difference in x and y coordinates for all pairs of keypoints
    dx = np.abs(keypoints2[:, :, 0] - keypoints1[:, :, 0])
    dy = np.abs(keypoints2[:, :, 1] - keypoints1[:, :, 1])
    # Compute the Manhattan distance for all pairs of keypoints
    distances = dx + dy
    return distances


def find_match(keypoints1, feature1, keypoints2, feature2):
    # find the match between two images
    # keypoints1: keypoints of image 1
    # feature1: feature of image 1
    # keypoints3: keypoints of image 3
    # feature3: feature of image 3
    # return the match keypoints of image 1 and image 3

    # compute manhattan distance between two keypoints
    distance = manhattan_distance(keypoints1, keypoints2)

    # compute the feature distance between two images
    feature_distance = np.linalg.norm(
        feature1[:, np.newaxis, :] - feature2[np.newaxis, :, :], axis=2
    )

    # compute the match
    weighted_distance = (distance < 20) * feature_distance + \
        (distance >= 20) * np.max(feature_distance)
    match1 = np.argmin(weighted_distance, axis=1)
    for i in range(len(keypoints1)):
        neighbor = distance[i] < 20
        if np.sum(neighbor) == 0:
            match1[i] = -1

    match2 = np.argmin(weighted_distance, axis=0)
    for i in range(len(keypoints2)):
        neighbor = distance[:, i] < 20
        if np.sum(neighbor) == 0:
            match2[i] = -1
    # convert to the format used in SuperGlue
    match_list_1 = []
    match_list_2 = []
    missing_list_1 = []
    missing_list_2 = []
    for i, match2_index in enumerate(match1):
        if match2_index == -1:
            missing_list_1.append(i)
        elif match2[match2_index] == i:
            match_list_1.append(i)
            match_list_2.append(match2_index)
        else:
            missing_list_1.append(i)
    for i, match1_index in enumerate(match2):
        if match1_index == -1:
            missing_list_2.append(i)
        elif match1[match1_index] == i:
            # match_list_2.append(i)
            pass
        else:
            missing_list_2.append(i)

    return match1, match2, match_list_1, match_list_2, missing_list_1, missing_list_2


def keypoints_cv2(keypoints):
    # convert the keypoint to cv2 keypoint
    cv2_keypoints = [
        cv2.KeyPoint(x=float(pt[1]), y=float(pt[0]), size=1) for pt in keypoints
    ]
    return cv2_keypoints


def visualize_keypoints(image, keypoints, save_path):
    cv2_keypoints = [
        cv2.KeyPoint(x=float(pt[1]), y=float(pt[0]), size=1) for pt in keypoints
    ]

    cv2_image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255
    cv2_image = cv2_image.astype(np.uint8)

    image_keypoints = cv2.drawKeypoints(np.squeeze(cv2_image, 0),
                                        cv2_keypoints, None,
                                        color=(0, 255, 0),
                                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite(save_path, image_keypoints)


def visualize_array(array, save_path):
    array = (array - np.min(array)) / (np.max(array) - np.min(array)) * 255
    array = array.astype(np.uint8)
    cv2.imwrite(save_path, array)


def label_to_keypoints(label, num_keypoints):
    keypoint_mask = []
    class_coordinates = []

    # Get the coordinates for each class using a list comprehension
    for class_label in range(num_keypoints):
        class_position = (np.argwhere(label[0] == class_label))
        if class_position.shape[0] == 0:
            keypoint_mask.append(0)
        elif class_position.shape[0] == 1:
            class_coordinates.append(class_position)
            keypoint_mask.append(1)
        elif class_position.shape[0] > 1:
            # Get the mean of the coordinates
            keypoint = np.mean(class_position, axis=0, keepdims=True,
                               dtype=np.int32)
            class_coordinates.append(keypoint)
            keypoint_mask.append(1)
        else:
            raise ValueError("Something went wrong")

    # Concatenate the coordinates into a single NumPy array
    coordinates_array = np.vstack(class_coordinates)
    keypoint_mask = np.array(keypoint_mask)
    return coordinates_array, keypoint_mask


def make_array_image(array):
    array = (array - np.min(array)) / (np.max(array) - np.min(array)) * 255
    array = array.astype(np.uint8)
    return array


def visualize_corespondence(image1, image2, keypoints1, keypoints2, save_path):
    matches_np = np.stack(
        [
            np.arange(keypoints1.shape[0]),
            np.arange(keypoints1.shape[0]),
            np.ones(keypoints1.shape[0])
        ],
        axis=1
    )
    matches_cv2 = [
        cv2.DMatch(int(queryIdx), int(trainIdx), 0, float(distance)) for queryIdx, trainIdx, distance in matches_np
    ]
    matched_image = cv2.drawMatches(
        image1[0], keypoints_cv2(keypoints1), image2[0], keypoints_cv2(keypoints2), matches_cv2, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    cv2.imwrite(save_path, matched_image)


def visulize_matches(image1, keypoints1, match1, image2, keypoints2, match2, save_path):

    matches_list = []
    for i, match in enumerate(match1):
        if match != -1:
            matches_list.append([i, match, 1])
    matches_np = np.array(matches_list)
    matches_cv2 = [
        cv2.DMatch(int(queryIdx), int(trainIdx), 0, float(distance)) for queryIdx, trainIdx, distance in matches_np
    ]
    matched_image = cv2.drawMatches(
        image1, keypoints_cv2(keypoints1), image2, keypoints_cv2(keypoints2), matches_cv2, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    cv2.imwrite(save_path, matched_image)


class CHD(Dataset):

    def __init__(self, keys, purpose, args):
        self.data_dir = args.data_dir
        self.patch_size = args.patch_size
        self.purpose = purpose
        self.classes = args.classes
        self.do_contrast = args.do_contrast
        self.files = []
        with open(os.path.join(self.data_dir, "mean_std.pkl"), 'rb') as f:
            mean_std = pickle.load(f)
        if self.do_contrast:
            # we do not pre-load all data, instead, load data in the get item function
            self.slice_position = []
            self.partition = []
            self.means = []
            self.stds = []
            self.total_frames = []
            for key in keys:
                frames = subfiles(
                    join(self.data_dir, 'train', key), False, None, ".npy", True)
                frames.sort()
                i = 0
                for frame in frames:
                    self.files.append(join(self.data_dir, 'train', key, frame))
                    self.means.append(mean_std[key]['mean'])
                    self.stds.append(mean_std[key]['std'])
                    self.slice_position.append(float(i+1)/len(frames))
                    part = len(frames) / 4.0
                    if part - int(part) >= 0.5:
                        part = int(part + 1)
                    else:
                        part = int(part)
                    self.partition.append(max(0, min(int(i//part), 3)+1))
                    self.total_frames.append(len(frames))
                    i = i + 1
        else:
            self.means = []
            self.stds = []
            self.total_frames = []
            for key in keys:
                frames = subfiles(
                    join(self.data_dir, 'train', 'ct_'+str(key)), False, None, ".npz", True)
                frames.sort()
                for frame in frames:
                    self.means.append(mean_std['ct_'+str(key)]['mean'])
                    self.stds.append(mean_std['ct_'+str(key)]['std'])
                    self.files.append(
                        join(self.data_dir, 'train', 'ct_'+str(key), frame)
                    )
                    self.total_frames.append(len(frames))
        print(f'dataset length: {len(self.files)}')

    def _get_neighbor_frame_path(self, index):
        # find a neighbor image
        current_file_path = self.files[index]
        current_frame_index = int(
            current_file_path.split('/')[-1].split('.')[0][5:])

        neighbor_frame_index = int(
            0.05 * random.uniform(-1, 1) * self.total_frames[index]) + current_frame_index
        while neighbor_frame_index == current_frame_index or neighbor_frame_index < 0 or neighbor_frame_index >= self.total_frames[index]:
            neighbor_frame_index = int(
                0.05 * random.uniform(-1, 1) * self.total_frames[index]) + current_frame_index
        if current_file_path.endswith('.npy'):
            npy_file_name = 'frame' + f'{neighbor_frame_index:03d}' + '.npy'
        else:
            npy_file_name = 'frame' + f'{neighbor_frame_index:03d}' + '.npz'
        # npy_file_name = 'frame' + f'{neighbor_frame_index:03d}' + '.npy'
        neighbor_file_path = current_file_path.split(
            '/')[:-1] + [npy_file_name]
        neighbor_file_path = '/'.join(neighbor_file_path)

        return neighbor_file_path

    def _contrast_process(self, image_path, debug=True):
        image = np.load(image_path).astype(np.float32)
        index = self.files.index(image_path)
        image -= self.means[index]
        image /= self.stds[index]
        keypoints, feature = gather_keypoints(image[None])
        if keypoints is None:
            return None, None, None, None, None, None
        # convert keypoints to label array
        label_array = np.ones_like(image) * (-1)
        label_array[keypoints[:, 0], keypoints[:, 1]] = np.arange(
            keypoints.shape[0]
        )
        if debug:
            visualize_keypoints(image[None], keypoints, 'keypoints.png')
        # resize image
        img, coord = pad_and_or_crop(image, self.patch_size, mode='random')
        label_array, _ = pad_and_or_crop(
            label_array, self.patch_size, mode='fixed', coords=coord
        )
        img1, img2, label1 = self._prepare_contrast_keypoints(
            img, label_array
        )
        # convert the label back to keypoints
        keypoints1, keypoints1_mask = label_to_keypoints(
            label1, num_keypoints=keypoints.shape[0]
        )
        keypoints1_mask = (keypoints1_mask == 1)
        if debug:
            visualize_keypoints(img1, keypoints1, 'keypoints1.png')
            visualize_corespondence(
                make_array_image(image[None]),
                make_array_image(img1),
                keypoints[keypoints1_mask], keypoints1,
                'correspondence.png'
            )

        return image, img1, img2, keypoints[keypoints1_mask], feature[keypoints1_mask], keypoints1

    def _sup_process(self, image, label):
        keypoints, _ = gather_keypoints(image[None])

        if keypoints is None:
            return None, None, None

        # convert keypoints to label array
        label_array = np.ones_like(image) * (-1)
        label_array[keypoints[:, 0], keypoints[:, 1]] = np.arange(
            keypoints.shape[0]
        )
        # resize image
        image, coord = pad_and_or_crop(image, self.patch_size, mode='random')
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

        return image[None], label[None], keypoints1

    def __getitem__(self, index):
        debug = False
        if self.do_contrast:
            img, img1, img2, kp, fea, kp1 = self._contrast_process(
                self.files[index]
            )
            if img is None:
                return None

            neighbor_file_path = self._get_neighbor_frame_path(index)
            nimg, nimg1, nimg2, nkp, nfea, nkp1 = self._contrast_process(
                neighbor_file_path
            )
            if nimg is None:
                return None

            match, nmatch, match_list_1, match_list_2, missing_list_1, missing_list_2 = find_match(
                kp, fea, nkp, nfea
            )

            # reorder the keypoints according to the match
            match_missing_1 = match_list_1 + missing_list_1
            reordered_kp1 = kp1[match_missing_1]
            match_missing_2 = match_list_2 + missing_list_2
            reordered_nkp1 = nkp1[match_missing_2]
            match = [i for i in range(len(match_list_1))] + \
                [-1 for i in range(len(missing_list_1))]
            nmatch = [i for i in range(len(match_list_2))] + \
                [-1 for i in range(len(missing_list_2))]

            if debug:
                visulize_matches(
                    make_array_image(img), kp[match_missing_1], match,
                    make_array_image(nimg), nkp[match_missing_2], nmatch,
                    'match.png'
                )
                visulize_matches(
                    make_array_image(img1[0]), reordered_kp1, match,
                    make_array_image(nimg1[0]), reordered_nkp1, nmatch,
                    'da_match.png'
                )
            # neighbor_image = np.load(neighbor_file_path).astype(np.float32)
            # neighbor_image -= self.means[index]
            # neighbor_image /= self.stds[index]
            # img3, img4 = self.prepare_contrast(neighbor_image)

            # find the correspondence between img1 and img3
            # keypoints1, feature1 = gather_keypoints(img1)
            # keypoints3, feature3 = gather_keypoints(img3)
            # match_index = find_match(keypoints1, feature1, keypoints3,
            #                          feature3)
            # visualize_keypoints(img1, keypoints1, 'Keypoints1.png')
            # visualize_keypoints(img3, keypoints3, 'Keypoints3.png')
            # return img1, nimg1, kp1, nkp1, match_list_1, match_list_2, missing_list_1, missing_list_2
            return img1, nimg1, reordered_kp1, reordered_nkp1, match, nmatch, self.slice_position[index]

            # return img1, img2, self.slice_position[index], self.partition[index]
        else:
            all_data = np.load(self.files[index])['data']
            img = all_data[0].astype(np.float32)
            img -= self.means[index]
            img /= self.stds[index]
            label = all_data[1].astype(np.float32)

            img, label, keypoints = self._sup_process(img, label)

            if img is None or label is None or keypoints is None:
                return None

            keypoints_label = np.squeeze(
                label, 0
            )[keypoints[:, 0], keypoints[:, 1]]

            neighbor_file_path = self._get_neighbor_frame_path(index)
            nimg, nlabel, nkeypoints, nkl = self.load_file(neighbor_file_path)
            while nimg is None:
                neighbor_file_path = self._get_neighbor_frame_path(index)
                nimg, nlabel, nkeypoints, nkl = self.load_file(
                    neighbor_file_path
                )

            return img, label, keypoints, keypoints_label, nimg, nkeypoints

    def load_file(self, file_path):
        all_data = np.load(file_path)['data']
        img = all_data[0].astype(np.float32)
        index = self.files.index(file_path)
        img -= self.means[index]
        img /= self.stds[index]
        label = all_data[1].astype(np.float32)

        img, label, keypoints = self._sup_process(img, label)

        if img is None or label is None or keypoints is None:
            return None, None, None, None

        keypoints_label = np.squeeze(
            label, 0
        )[keypoints[:, 0], keypoints[:, 1]]

        return img, label, keypoints, keypoints_label

    # this function for normal supervised training

    def prepare_supervised(self, img, label):
        if self.purpose == 'train':
            # pad image
            img, coord = pad_and_or_crop(img, self.patch_size, mode='random')
            label, _ = pad_and_or_crop(
                label, self.patch_size, mode='fixed', coords=coord)
            # No augmentation is used in the finetuning because augmention could hurt the performance.
            return img[None], label[None]

        else:
            # resize image
            img, coord = pad_and_or_crop(img, self.patch_size, mode='centre')
            label, _ = pad_and_or_crop(
                label, self.patch_size, mode='fixed', coords=coord)
            return img[None], label[None]

    def _prepare_contrast_keypoints(self, img, label):
        # the image and label should be [batch, c, x, y, z], this is the adapatation for using batchgenerators :)
        data_dict = {'data': img[None, None], 'seg': label[None, None]}
        tr_transforms = []
        tr_transforms.append(MirrorTransform((0, 1)))
        tr_transforms.append(
            RndTransform(
                SpatialTransform(
                    self.patch_size,
                    list(np.array(self.patch_size)//2),
                    True, (100., 350.), (14., 17.),
                    True,
                    (0, 2.*np.pi), (-0.000001, 0.00001), (-0.000001, 0.00001),
                    True, (0.7, 1.3), 'constant', 0, 3, 'constant', 0, 0,
                    random_crop=False
                ),
                prob=0.67,
                alternative_transform=RandomCropTransform(self.patch_size)
            )
        )

        train_transform = Compose(tr_transforms)
        data_dict1 = train_transform(**data_dict)
        img1 = data_dict1.get('data')[0]

        label1 = data_dict1.get('seg')[0]

        data_dict2 = train_transform(**data_dict)
        img2 = data_dict2.get('data')[0]
        return img1, img2, label1

    # use this function for contrastive learning
    def prepare_contrast(self, img):

        # the image and label should be [batch, c, x, y, z], this is the adapatation for using batchgenerators :)
        data_dict = {'data': img[None, None]}
        tr_transforms = []
        tr_transforms.append(MirrorTransform((0, 1)))
        tr_transforms.append(
            RndTransform(
                SpatialTransform(
                    self.patch_size,
                    list(np.array(self.patch_size)//2),
                    True, (100., 350.), (14., 17.),
                    True,
                    (0, 2.*np.pi), (-0.000001, 0.00001), (-0.000001, 0.00001),
                    True, (0.7, 1.3), 'constant', 0, 3, 'constant', 0, 0,
                    random_crop=False
                ),
                prob=0.67,
                alternative_transform=RandomCropTransform(self.patch_size)
            )
        )

        train_transform = Compose(tr_transforms)
        data_dict1 = train_transform(**data_dict)
        img1 = data_dict1.get('data')[0]
        data_dict2 = train_transform(**data_dict)
        img2 = data_dict2.get('data')[0]
        return img1, img2

    def __len__(self):
        return len(self.files)


def get_split_chd(data_dir, fold, seed=12345):
    # this is seeded, will be identical each time
    all_keys = np.arange(0, 50)
    cases = os.listdir(data_dir)
    cases.sort()
    i = 0
    for case in cases:
        all_keys[i] = int(case[-4:])
        i = i + 1
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    splits = kf.split(all_keys)
    for i, (train_idx, test_idx) in enumerate(splits):
        train_keys = all_keys[train_idx]
        test_keys = all_keys[test_idx]
        if i == fold:
            break
    return train_keys, test_keys


def chd_sg_sup_collate(batch):
    if None in batch:
        print('None in batch')
        batch = [item for item in batch if item is not None]

    # batch = list(filter(lambda x: x is not None, batch))
    num_keypoints = min([len(item[2]) for item in batch])
    neighbor_num_keypoints = min([len(item[5]) for item in batch])
    num_keypoints = min(num_keypoints, neighbor_num_keypoints)
    batch = [list(item) for item in batch]
    for i in range(len(batch)):
        selected_rows = np.random.choice(
            batch[i][2].shape[0], size=num_keypoints, replace=False
        )
        batch[i][2] = batch[i][2][selected_rows]
        batch[i][3] = batch[i][3][selected_rows]

        selected_rows = np.random.choice(
            batch[i][5].shape[0], size=num_keypoints, replace=False
        )
        batch[i][5] = batch[i][5][selected_rows]

    return torch.utils.data.dataloader.default_collate(batch)


def chd_sg_collate(batch):
    if None in batch:
        print('None in batch')
        batch = [item for item in batch if item is not None]

    # with each batch
    # img1, nimg1, kp1, nkp1, match_list_1, match_list_2, missing_list_1, missing_list_2
    images_1 = [torch.from_numpy(item[0]) for item in batch]
    images_1 = torch.stack(images_1, dim=0)

    images_2 = [torch.from_numpy(item[1]) for item in batch]
    images_2 = torch.stack(images_2, dim=0)

    slice_position = [torch.tensor(item[6]) for item in batch]
    slice_position = torch.stack(slice_position, dim=0)

    min_num_keypoints = min([len(item[2]) for item in batch])
    min_num_keypoints_neighbor = min([len(item[3]) for item in batch])
    num_keypoints = min(min_num_keypoints, min_num_keypoints_neighbor)

    keypoints_1 = []
    keypoints_2 = []

    all_match_index_0 = torch.empty(0, dtype=torch.int64)
    all_match_index_1 = torch.empty(0, dtype=torch.int64)
    all_match_index_2 = torch.empty(0, dtype=torch.int64)

    matches_list = []

    for i in range(len(batch)):
        kp1 = batch[i][2]
        nkp1 = batch[i][3]
        match1 = batch[i][4]
        nmatch1 = batch[i][5]

        trucated_kp1 = kp1[:num_keypoints]
        trucated_nkp1 = nkp1[:num_keypoints]
        trucated_match1 = match1[:num_keypoints]
        trucated_nmatch1 = nmatch1[:num_keypoints]
        trucated_match1 = np.array(trucated_match1, dtype=np.int32)
        trucated_nmatch1 = np.array(trucated_nmatch1, dtype=np.int32)
        trucated_match1[trucated_match1 >= num_keypoints] = -1
        trucated_nmatch1[trucated_nmatch1 >= num_keypoints] = -1

        match_list_1 = []
        match_list_2 = []
        missing_list_1 = []
        missing_list_2 = []
        for j, match2_index in enumerate(trucated_match1):
            if match2_index == -1:
                missing_list_1.append(j)
            elif trucated_nmatch1[match2_index] == j:
                match_list_1.append(j)
                match_list_2.append(match2_index)
            else:
                missing_list_1.append(j)
        for j, match2_index in enumerate(trucated_nmatch1):
            if match2_index == -1:
                missing_list_2.append(j)
            elif trucated_match1[match2_index] == j:
                pass
            else:
                missing_list_2.append(j)

        match_list_1 = torch.from_numpy(np.array(match_list_1, dtype=np.int32))
        match_list_2 = torch.from_numpy(np.array(match_list_2, dtype=np.int32))
        missing_list_1 = torch.from_numpy(
            np.array(missing_list_1, dtype=np.int32)
        )
        missing_list_2 = torch.from_numpy(
            np.array(missing_list_2, dtype=np.int32)
        )

        keypoints_1.append(torch.from_numpy(trucated_kp1))
        keypoints_2.append(torch.from_numpy(trucated_nkp1))

        match_index_0 = torch.empty(
            len(match_list_1) + len(missing_list_1) + len(missing_list_2), dtype=torch.long
        ).fill_(i)
        all_match_index_0 = torch.cat(
            [
                all_match_index_0,
                torch.empty(
                    len(match_list_1) + len(missing_list_1) + len(missing_list_2), dtype=torch.long
                ).fill_(i)
            ]
        )
        match_index_1 = torch.cat(
            [
                match_list_1, missing_list_1,
                torch.empty(
                    len(missing_list_2), dtype=torch.long
                ).fill_(-1)
            ]
        )
        all_match_index_1 = torch.cat(
            [
                all_match_index_1, match_list_1, missing_list_1,
                torch.empty(
                    len(missing_list_2), dtype=torch.long
                ).fill_(-1)
            ]
        )
        match_index_2 = torch.cat(
            [
                match_list_2,
                torch.empty(
                    len(missing_list_1), dtype=torch.long
                ).fill_(-1),
                missing_list_2
            ]
        )
        all_match_index_2 = torch.cat(
            [
                all_match_index_2, match_list_2,
                torch.empty(
                    len(missing_list_1), dtype=torch.long
                ).fill_(-1),
                missing_list_2
            ]
        )

        item_matches = torch.stack(
            [match_index_0, match_index_1, match_index_2], -1
        )
        matches_list.append(item_matches)

    match_indexes = torch.stack(
        [all_match_index_0, all_match_index_1, all_match_index_2], -1
    )
    gt_vector = torch.ones(len(match_indexes), dtype=torch.float32)

    keypoints_1 = torch.stack(keypoints_1, dim=0)
    keypoints_2 = torch.stack(keypoints_2, dim=0)

    return {
        'keypoints_1': keypoints_1, 'keypoints_2': keypoints_2,
        'images_1': images_1, 'images_2': images_2,
        'matches': match_indexes,
        'matches_list': matches_list,
        'batch_index': torch.arange(len(batch), dtype=torch.long),
        'gt_vec': gt_vector,
        'slice_position': slice_position,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str,
                        default="/afs/crc.nd.edu/user/d/dzeng2/data/chd/preprocessed_without_label/")
    parser.add_argument("--patch_size", type=tuple, default=(512, 512))
    parser.add_argument("--classes", type=int, default=8)
    parser.add_argument("--do_contrast", default=True, action='store_true')
    parser.add_argument("--slice_threshold", type=float, default=0.05)
    args = parser.parse_args()

    train_keys = os.listdir(os.path.join(args.data_dir, 'train'))
    train_keys.sort()
    train_dataset = CHD(keys=train_keys, purpose='train', args=args)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=30,
                                                   shuffle=True,
                                                   num_workers=8,
                                                   drop_last=False)

    pp = []
    n = 0
    for batch_idx, tup in enumerate(train_dataloader):
        print(f'the {n}th minibatch...')
        img1, img2, slice_position, partition = tup
        batch_size = img1.shape[0]
        # print(f'batch_size:{batch_size}, slice_position:{slice_position}')
        slice_position = slice_position.contiguous().view(-1, 1)
        mask = (torch.abs(slice_position.T.repeat(batch_size, 1) -
                slice_position.repeat(1, batch_size)) < args.slice_threshold).float()
        # count how many positive pair in each batch
        for i in range(mask.shape[0]):
            pp.append(mask[i].sum()-1)
        n = n + 1
        if n > 100:
            break
    pp = np.asarray(pp)
    pp_mean = np.mean(pp)
    pp_std = np.std(pp)
    print(f'mean:{pp_mean}, std:{pp_std}')
