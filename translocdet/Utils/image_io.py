from typing import Tuple

import nd2reader as nd
import numpy as np


def read_input_image_separate(input_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    image_os = nd.ND2Reader(input_file)
    channel0 = image_os[0]
    channel1 = image_os[1]
    channel2 = image_os[2]

    return channel0, channel1, channel2


def read_input_image_collate(input_file: str) -> np.ndarray:
    image_os = nd.ND2Reader(input_file)
    channel0 = image_os[0]
    channel1 = image_os[1]
    channel2 = image_os[2]

    new_image = np.zeros(channel0.shape + (3,))
    new_image[:, :, 0] = channel0
    new_image[:, :, 1] = channel1
    new_image[:, :, 2] = channel2

    return new_image
