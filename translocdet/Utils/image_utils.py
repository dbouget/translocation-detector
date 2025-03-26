import numpy as np


def normalize_image(input_image: np.ndarray) -> np.ndarray:
    """
    Adjust the input image to have an intensity range between 0 and 255 (uint8).

    :param input_image: Image to be normalized.
    :return: Normalized image.
    """
    res = np.zeros(input_image.shape)
    if len(input_image.shape) == 2:
        min = np.min(input_image)
        max = np.max(input_image - min)
        res = ((input_image - min) / max) * 255.0
    elif len(input_image.shape) == 3:
        for c in range(input_image.shape[2]):
            min = np.min(input_image[..., c])
            max = np.max(input_image[..., c] - min)
            tmp = ((input_image[..., c] - min) / max) * 255.0
            res[..., c] = tmp

    return res


def rgb2gray(rgb: np.ndarray) -> np.ndarray:
    """
    Performs color conversion on the input RGB image to Grayscale.
    :param rgb: Input RGB image to color convert.
    :return: Grayscale version of the input RGB image.
    """
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140]).astype("uint8")


def is_region_matching_keypoint(region, keypoint_list) -> bool:
    """

    :param region:
    :param keypoint_list:
    :return:
    """
    res = False
    bbox = region.bbox
    # centroid = region.centroid
    for kp in keypoint_list:
        if bbox[0] < kp.pt[1] < bbox[2] and bbox[1] < kp.pt[0] < bbox[3]:
            return True
    return res


def get_bbox_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Returns
    -------
    float
        Value in [0, 1]
    """
    assert bb1[0] < bb1[2]
    assert bb1[1] < bb1[3]
    assert bb2[0] < bb2[2]
    assert bb2[1] < bb2[3]

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou
