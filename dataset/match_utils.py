import cv2
import numpy as np
from batchgenerators.transforms.abstract_transforms import (Compose,
                                                            RndTransform)
from batchgenerators.transforms.crop_and_pad_transforms import \
    RandomCropTransform
from batchgenerators.transforms.spatial_transforms import (MirrorTransform,
                                                           SpatialTransform)

from .utils import pad_and_or_crop


def make_array_image(array):
    array = (array - np.min(array)) / (np.max(array) - np.min(array)) * 255
    array = array.astype(np.uint8)
    return array


def gather_keypoints(image):
    assert image.ndim == 2

    # normalize image
    image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255
    image = image.astype(np.uint8)

    # detect keypoints
    sift = cv2.SIFT_create()
    keypoints, features = sift.detectAndCompute(image, None)

    # convert keypoints to numpy array
    # the order is y, x
    keypoints = np.array([[kp.pt[1], kp.pt[0]] for kp in keypoints])
    keypoints = np.round(keypoints).astype(np.int32)

    # conner case
    if keypoints.shape[0] == 0:
        return None, None

    # random permute the keypoints and features
    perm = np.random.permutation(len(keypoints))
    keypoints = keypoints[perm]
    features = features[perm]

    return keypoints, features


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


def prepare_transform(patch_size):
    tr_transforms = []
    tr_transforms.append(MirrorTransform((0, 1)))
    tr_transforms.append(
        RndTransform(
            SpatialTransform(
                patch_size,
                list(np.array(patch_size)//2),
                True, (100., 350.), (14., 17.),
                True, (0, 2.*np.pi),
                (-0.000001, 0.00001), (-0.000001, 0.00001),
                True, (0.7, 1.3),
                'constant', 0, 3, 'constant', 0, 0,
                random_crop=False
            ),
            prob=0.67,
            alternative_transform=RandomCropTransform(patch_size)
        )
    )
    transform = Compose(tr_transforms)
    return transform


def augment_image_and_label(image, label, transform):
    # the image and label should be [batch, c, x, y, z], this is the adapatation for using batchgenerators :)
    data_dict = {'data': image[None, None], 'seg': label[None, None]}

    data_dict = transform(**data_dict)

    augmented_image = data_dict.get('data')[0]
    augmented_label = data_dict.get('seg')[0]

    return augmented_image, augmented_label


def get_two_augmentation(img, label, patch_size):

    train_transform = prepare_transform(patch_size)

    img1, label1 = augment_image_and_label(img, label, train_transform)
    img2, label2 = augment_image_and_label(img, label, train_transform)

    return img1, img2, label1, label2


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
            keypoint = np.mean(
                class_position, axis=0, keepdims=True, dtype=np.int32
            )
            class_coordinates.append(keypoint)
            keypoint_mask.append(1)
        else:
            raise ValueError("Something went wrong")

    if len(class_coordinates) == 0:
        return None, None

    # Concatenate the coordinates into a single NumPy array
    coordinates_array = np.vstack(class_coordinates)
    keypoint_mask = np.array(keypoint_mask)

    return coordinates_array, keypoint_mask


def keypoints_cv2(keypoints):
    # convert the keypoint to cv2 keypoint
    cv2_keypoints = [
        cv2.KeyPoint(x=float(pt[1]), y=float(pt[0]), size=1) for pt in keypoints
    ]
    return cv2_keypoints


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


def contrast_process(
    image, patch_size, debug=False, mean=None, std=None,
):
    # normalize image
    if mean is not None:
        image -= mean
    if std is not None:
        image /= std

    # gather keypoints
    keypoints, feature = gather_keypoints(image)

    # conner case
    if keypoints is None or feature is None:
        return None, None, None, None, None, None

    # convert keypoints to label array
    label_array = np.ones_like(image) * (-1)
    label_array[keypoints[:, 0], keypoints[:, 1]] = np.arange(
        keypoints.shape[0]
    )

    # visualize keypoints
    if debug:
        visualize_keypoints(image, keypoints, 'keypoints.png')

    # resize image
    img, coord = pad_and_or_crop(image, patch_size, mode='random')
    label_array, _ = pad_and_or_crop(
        label_array, patch_size, mode='fixed', coords=coord
    )
    img1, img2, label1, label2 = get_two_augmentation(
        img, label_array, patch_size
    )

    # convert the label back to keypoints
    keypoints1, keypoints1_mask = label_to_keypoints(
        label1, num_keypoints=keypoints.shape[0]
    )
    if keypoints1 is None or keypoints1_mask is None:
        return None, None, None, None, None, None

    keypoints1_mask = (keypoints1_mask == 1)

    # visualize keypoints
    if debug:
        visualize_keypoints(img1[0], keypoints1, 'keypoints1.png')
        visualize_corespondence(
            make_array_image(image[None]),
            make_array_image(img1),
            keypoints[keypoints1_mask], keypoints1,
            'correspondence.png'
        )

    return image, img1, img2, keypoints[keypoints1_mask], feature[keypoints1_mask], keypoints1


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
    weighted_distance = (distance < 30) * feature_distance + \
        (distance >= 30) * np.max(feature_distance)
    match1 = np.argmin(weighted_distance, axis=1)
    for i in range(len(keypoints1)):
        neighbor = distance[i] < 20
        if np.sum(neighbor) == 0:
            match1[i] = -1

    match2 = np.argmin(weighted_distance, axis=0)
    for i in range(len(keypoints2)):
        neighbor = distance[:, i] < 30
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


def gather_match_information(
    image, neighbor_image, patch_size, debug=False
):
    # print('working on it')
    img, img1, img2, kp, fea, kp1 = contrast_process(
        image, patch_size
    )
    if (kp1 is None) or (img1 is None) or (kp is None):
        return None, None, None, None, None, None

    nimg, nimg1, nimg2, nkp, nfea, nkp1 = contrast_process(
        neighbor_image, patch_size
    )
    if (nkp1 is None) or (nimg1 is None) or (nkp is None):
        return None, None, None, None, None, None

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

    # print('finish working and return')

    return img1, nimg1, reordered_kp1, reordered_nkp1, match, nmatch
