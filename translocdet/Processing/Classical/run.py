import logging
import os.path
from copy import deepcopy
from typing import List
from typing import Tuple

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.morphology import binary_closing
from skimage.measure import label
from skimage.measure import regionprops
from skimage.morphology import skeletonize

from translocdet.Utils.configuration import ResourcesConfiguration
from translocdet.Utils.image_io import read_input_image_collate
from translocdet.Utils.image_io import read_input_image_separate
from translocdet.Utils.image_utils import get_bbox_iou
from translocdet.Utils.image_utils import is_region_matching_keypoint
from translocdet.Utils.image_utils import normalize_image
from translocdet.Utils.image_utils import rgb2gray


def run_classical():
    """

    :return:
    """
    try:
        user_input = ResourcesConfiguration.getInstance().system_input_path
        output_folder = ResourcesConfiguration.getInstance().system_output_folder
        os.makedirs(output_folder, exist_ok=True)

        if os.path.exists(user_input) and not os.path.isdir(user_input):
            processing(input_file=user_input, output_folder=output_folder)
        elif os.path.isdir(user_input):
            input_images = []
            for _, _, files in os.walk(user_input):
                for f in files:
                    if f.endswith(tuple(ResourcesConfiguration.getInstance().accepted_image_formats)):
                        input_images.append(os.path.join(user_input, f))
                break

            for img_fn in input_images:
                processing(input_file=img_fn, output_folder=output_folder)
    except Exception as e:
        raise ValueError(f"Classical computation failed with {e}")


def processing(input_file: str, output_folder: str) -> None:
    """
    Main classical processing techniques working in three steps: (i) identification of the image region showing
    chromosomes from an open cell to crop the minimal Region Of Interest (ROI), (ii) detection of all individual
    chromosomes visible in the ROI, (iii) identification of chromosomes with activated receptors (i.e., red and
    green end-points) and where potential translocation happened.
    :param input_file: Full path to the original image to analyze.
    :param output_folder: Full path to the folder where the processing results will be saved.
    :return:
    """
    # Creating destination folders
    res_dir = os.path.join(output_folder, os.path.basename(input_file).split(".")[0])
    os.makedirs(res_dir, exist_ok=True)
    color_res_dir = os.path.join(res_dir, "ColorWork")
    os.makedirs(color_res_dir, exist_ok=True)

    try:
        # Opening the input image from file on disk to np.ndarray
        input_c0, input_c1, input_c2 = read_input_image_separate(input_file)
        input_collate = read_input_image_collate(input_file)

        input_c0_norm = normalize_image(input_c0).astype("uint8")
        input_c1_norm = normalize_image(input_c1).astype("uint8")
        input_c2_norm = normalize_image(input_c2).astype("uint8")
        input_collated_norm = normalize_image(input_collate).astype("uint8")

        # Viewing
        fig = plt.figure()
        fig.suptitle("Per channel input image")
        fig.add_subplot(2, 2, 1)
        plt.imshow(input_c0_norm)
        fig.add_subplot(2, 2, 2)
        plt.imshow(input_c1_norm)
        fig.add_subplot(2, 2, 3)
        plt.imshow(input_c2_norm)
        fig.add_subplot(2, 2, 4)
        plt.imshow(input_collated_norm)
        plt.savefig(os.path.join(color_res_dir, "per_channel_input.png"), bbox_inches="tight", dpi=300)
        plt.close(fig)
        high_marker_color = np.zeros(input_c0_norm.shape).astype("uint8")
        high_marker_color[np.logical_or(input_c1_norm > 180, input_c2_norm > 180)] = 255
        cv.imwrite(os.path.join(color_res_dir, "green_and_red_in_whole_image.png"), high_marker_color)

        # Identification of the non-opened cells and of the final ROI.
        bbox = remove_cells(input_collated_norm.astype("uint8"), high_marker_color, res_dir)

        # Viewing
        cv.imwrite(
            os.path.join(res_dir, "roi_selection.png"),
            cv.rectangle(input_collated_norm.astype("uint8"), (bbox[1], bbox[0]), (bbox[3], bbox[2]), (0, 255, 0), 2),
        )
        input_c0_roi = input_c0[bbox[0] : bbox[2], bbox[1] : bbox[3]]
        input_c1_roi = input_c1[bbox[0] : bbox[2], bbox[1] : bbox[3]]
        input_c2_roi = input_c2[bbox[0] : bbox[2], bbox[1] : bbox[3]]
        input_c0_roi_norm = normalize_image(input_c0_roi).astype("uint8")
        input_c1_roi_norm = normalize_image(input_c1_roi).astype("uint8")
        input_c2_roi_norm = normalize_image(input_c2_roi).astype("uint8")
        input_collate_roi = np.zeros(input_c0_roi.shape + (3,))
        input_collate_roi[..., 0] = input_collate[..., 0][bbox[0] : bbox[2], bbox[1] : bbox[3]]
        input_collate_roi[..., 1] = input_collate[..., 1][bbox[0] : bbox[2], bbox[1] : bbox[3]]
        input_collate_roi[..., 2] = input_collate[..., 2][bbox[0] : bbox[2], bbox[1] : bbox[3]]
        input_collate_roi_norm = normalize_image(input_collate_roi).astype("uint8")
        fig = plt.figure()
        fig.suptitle("Per channel image ROI")
        fig.add_subplot(2, 2, 1)
        plt.imshow(input_c0_roi_norm)
        fig.add_subplot(2, 2, 2)
        plt.imshow(input_c1_roi_norm)
        fig.add_subplot(2, 2, 3)
        plt.imshow(input_c2_roi_norm)
        fig.add_subplot(2, 2, 4)
        plt.imshow(input_collate_roi_norm)
        plt.savefig(os.path.join(res_dir, "roi_selection_channelwise.png"), bbox_inches="tight", dpi=300)
        plt.close(fig)
        high_marker_color_roi = np.zeros(input_c0_roi_norm.shape).astype("uint8")
        high_marker_color_roi[np.logical_or(input_c1_roi_norm > 180, input_c2_roi_norm > 180)] = 255
        cv.imwrite(os.path.join(color_res_dir, "green_and_red_in_roi.png"), high_marker_color_roi)

        # Detection of individual chromosomes and identification of targeted chromosomes
        chromosome_markers = detect_chromosomes(input_collate_roi_norm.astype("uint8"), high_marker_color_roi, res_dir)
        cr_bboxes, cr_numbers, sus_cr_numbers = identify_tagged_chromosomes(
            input_collate_roi_norm.astype("uint8"),
            chromosome_markers,
            input_c0_roi_norm,
            input_c1_roi_norm,
            input_c2_roi_norm,
            res_dir,
        )
        print("Identified chromosomes: {}".format(cr_numbers))
        print("Suspicious chromosomes: {}".format(sus_cr_numbers))

        # Saving the final translocation report on disk
        report_fn = os.path.join(res_dir, "translocation_report.txt")
        with open(report_fn, "w") as f:
            f.write(f"Translocation report for {os.path.basename(input_file).split('.')[0]}:\n\n")
            f.write(f"Identified chromosomes: {cr_numbers}\n")
            f.write(f"Suspicious chromosomes: {sus_cr_numbers}\n")
        f.close()
    except Exception as e:
        raise ValueError(f"Main processing in classical computation failed with {e}")


def remove_cells(input_image: np.ndarray, marker_mask: np.ndarray, output_folder: str) -> List[int]:
    """
    Identification of the region of interest around the lying chromosomes from an opened cell by detecting all
    non-opened cells to be excluded from the final ROI computation.

    :param input_image: Original chromosome image unedited.
    :param marker_mask:
    :param output_folder: Full path of location on disk where the results should be saved.
    :return:
    """
    res_dir = os.path.join(output_folder, "Processing-Blobs")
    os.makedirs(res_dir, exist_ok=True)

    try:
        params = cv.SimpleBlobDetector_Params()
        params.filterByColor = True
        params.blobColor = 255
        params.filterByArea = True
        params.minArea = 4000
        params.maxArea = 100000
        params.filterByCircularity = True
        params.minCircularity = ResourcesConfiguration.getInstance().classical_blobdetector_min_circularity  # 0.1
        params.filterByConvexity = True
        params.minConvexity = ResourcesConfiguration.getInstance().classical_blobdetector_min_convexity  # 0.3
        params.filterByInertia = False

        gray_input = rgb2gray(input_image)
        input_image = np.stack(
            [np.pad(input_image[..., x], pad_width=50, mode="edge") for x in range(input_image.shape[2])], axis=-1
        )
        gray_input = np.pad(gray_input, pad_width=50, mode="edge")
        marker_mask_pad = np.pad(marker_mask, pad_width=50, mode="edge")
        ret, thresh = cv.threshold(gray_input, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

        # noise removal
        kernel = np.ones((3, 3), np.uint8)
        opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)
        # cv.imshow('Opening', opening)
        cv.imwrite(os.path.join(res_dir, "rmcells_opening.png"), opening)

        detector = cv.SimpleBlobDetector_create(params)
        keypoints = detector.detect(opening)
        img_with_keypoints = cv.drawKeypoints(
            input_image, keypoints, np.array([]), (0, 0, 255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        # cv.imshow('Blob Detection', img_with_keypoints)
        cv.imwrite(os.path.join(res_dir, "rmcells_blobs.png"), img_with_keypoints)
        mask_labelled = label(opening)
        # cv.imshow('Opening labelled', mask_labelled.astype('uint8'))
        cv.imwrite(os.path.join(res_dir, "rmcells_labelled.png"), mask_labelled.astype("uint8"))
        mask_labelled_regions = regionprops(mask_labelled)
        final_mask = deepcopy(opening)
        for r in mask_labelled_regions:
            state = is_region_matching_keypoint(region=r, keypoint_list=keypoints)
            border_object = 0 in r.bbox or input_image.shape[0] == r.bbox[2] or input_image.shape[1] == r.bbox[3]
            if state or border_object:
                final_mask[mask_labelled == r.label] = 0

        cv.imwrite(os.path.join(res_dir, "rmcells_blob_removed_mask.png"), final_mask)

        # large marker spots removed as most likely not being chromosome-related marker
        marker_mask_labelled = label(marker_mask_pad)
        marker_mask_regions = regionprops(marker_mask_labelled)

        # @TODO. Should compute the center of mass independently for each structure and then
        # get the average value of all to decide about a cut-off.
        mask_labelled = label(final_mask)
        mask_labelled_regions = regionprops(mask_labelled)
        all_coms = []
        for r in mask_labelled_regions:
            all_coms.append(r.centroid)
        average_com = np.average(np.asarray(all_coms), axis=0)
        # com = center_of_mass(final_mask)
        for r in mask_labelled_regions:
            bbox_centroid = r.centroid
            euclidean_dist = abs(bbox_centroid[0] - average_com[0]) + abs(bbox_centroid[1] - average_com[1])
            if euclidean_dist > 300:
                final_mask[mask_labelled == r.label] = 0
            for mr in marker_mask_regions:
                iou = get_bbox_iou(mr.bbox, r.bbox)
                if iou > 0.5:
                    final_mask[mask_labelled == r.label] = 0

        # cv.imshow('Final Mask', final_mask)
        cv.imwrite(os.path.join(res_dir, "rmcells_final_mask.png"), final_mask)

        i, j = np.where(final_mask == np.min(final_mask[np.nonzero(final_mask)]))
        bbox = [np.min(i), np.min(j), np.max(i), np.max(j)]
        # cv.imshow("selection", cv.rectangle(input_image, (bbox[1], bbox[0]), (bbox[3], bbox[2]), (0, 255, 0), 2))
        # cv.imwrite(os.path.join(res_dir, "rmcells_roi_selection.png"),
        # cv.rectangle(input_image, (bbox[1], bbox[0]), (bbox[3], bbox[2]), (0, 255, 0), 2))
        # cv.waitKey(0)

        return [bbox[0] - 50 - 25, bbox[1] - 50 - 25, bbox[2] - 50 + 25, bbox[3] - 50 + 25]
    except Exception as e:
        raise ValueError(f"ROI detection process failed with: {e}")


def detect_chromosomes(input_image: np.ndarray, marker_mask: np.ndarray, output_folder: str):
    """
    Performs a watershed analysis to identify all unique chromosome instances
    :param input_image:
    :param marker_mask:
    :param output_folder:
    :return:
    """
    # Destination folder creation
    res_dir = os.path.join(output_folder, "Processing-Instances")
    os.makedirs(res_dir, exist_ok=True)

    try:
        # Image color conversion
        process_ori_image = deepcopy(input_image)
        gray_input = rgb2gray(process_ori_image)
        kernel = np.ones((3, 3), np.uint8)

        # @TODO. Not working as intended, meant to remove the part of cell or bright spots still visible in the ROI.
        marker_mask[marker_mask == 255] = 1
        marker_mask_refined = binary_closing(marker_mask, iterations=1, structure=np.ones((5, 5), np.uint8))
        marker_mask_labelled = label(marker_mask_refined)
        marker_mask_regions = regionprops(marker_mask_labelled)
        for mr in marker_mask_regions:
            if mr.area > 450:
                gray_input[marker_mask_labelled == mr.label] = 0

        cv.imwrite(os.path.join(res_dir, "idchrom_cleaned_input.png"), gray_input)
        ret, thresh = cv.threshold(gray_input, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        # cv.imshow('Threshold', thresh)

        # noise removal
        opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=5)
        # cv.imshow('Opening', opening)
        cv.imwrite(os.path.join(res_dir, "idchrom_opening.png"), opening)

        # sure background area
        sure_bg = cv.dilate(opening, kernel, iterations=3)

        # Finding sure foreground area
        dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
        # @TODO. Have to come up with a way to clip the highest value to not mess up the distance over the chromosomes
        # if there's an artifact or bubble still visible.
        dist_transform[dist_transform > np.percentile(dist_transform, 99)] = np.percentile(dist_transform, 99)
        cv.normalize(dist_transform, dist_transform, 0, 1, cv.NORM_MINMAX)
        # cv.imshow('Dist transform', dist_transform)
        cv.imwrite(os.path.join(res_dir, "idchrom_distances.png"), dist_transform * 255)
        ret, sure_fg = cv.threshold(dist_transform, 0.75 * dist_transform.max(), 255, 0)
        skeleton_fg = skeletonize(sure_fg)
        plt.imsave(os.path.join(res_dir, "idchrom_skeleton.png"), skeleton_fg)

        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        sure_fg = cv.morphologyEx(sure_fg, cv.MORPH_DILATE, kernel, iterations=1)
        unknown = cv.subtract(sure_bg, sure_fg)
        # cv.imshow('Sure foreground', sure_fg)
        # cv.imshow('Sure background', sure_bg)
        # cv.imshow('Unknown background', unknown)
        cv.imwrite(os.path.join(res_dir, "idchrom_sure_foreground.png"), sure_fg)
        cv.imwrite(os.path.join(res_dir, "idchrom_sure_background.png"), sure_bg)
        cv.imwrite(os.path.join(res_dir, "idchrom_unknown_region.png"), unknown)

        # Marker labelling
        ret, markers = cv.connectedComponents(sure_fg)
        # Add one to all labels so that sure background is not 0, but 1
        markers = markers + 1
        # Now, mark the region of unknown with zero
        markers[unknown == 255] = 0
        markers = cv.watershed(process_ori_image, markers)
        process_ori_image[markers == -1] = [0, 255, 255]
        # cv.imshow('Markers', markers.astype('uint8'))
        # cv.imshow('Chromosomes', process_ori_image)
        cv.imwrite(os.path.join(res_dir, "idchrom_markers.png"), markers.astype("uint8"))
        cv.imwrite(os.path.join(res_dir, "idchrom_chromosomes.png"), process_ori_image)
        # cv.waitKey(0)

        return markers
    except Exception as e:
        raise ValueError(f"Chromosomes identification process failed with: {e}")


def identify_tagged_chromosomes(
    input_image: np.ndarray,
    markers: np.ndarray,
    input_c0: np.ndarray,
    input_c1: np.ndarray,
    input_c2: np.ndarray,
    output_folder: str,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Identifies from all detected chromosomes the ones that are activated (i.e., with red and green ends), and in such
    case of translocation the afflicted chromosomes (by their number).
    :param input_image:
    :param markers:
    :param input_c0:
    :param input_c1:
    :param input_c2:
    :param output_folder:
    :return: List of bounding box coordinates around activated chromosomes, list of activated chromosomes numbers, and
    list of activated chromosomes numbers with suspected translocation.
    """
    res_chrom_dir = os.path.join(output_folder, "Chromosomes")
    os.makedirs(res_chrom_dir, exist_ok=True)
    res_chrom_view_dir = os.path.join(output_folder, "Chromosomes-Views")
    os.makedirs(res_chrom_view_dir, exist_ok=True)

    try:
        nb_markers = np.unique(markers)
        highlighted_cr_bboxes = []
        highlighted_cr_numbers = []
        suspicious_cr_numbers = []
        for m in nb_markers[2:]:
            try:
                # chromosome_img = deepcopy(input_image)
                # chromosome_img[markers != m] = 0
                # chromosome_img_gray = rgb2gray(chromosome_img)
                # plt.imshow(chromosome_img_gray, cmap='gray')
                # plt.show()
                kernel = np.ones((3, 3), np.uint8)
                chromosome_img_mask = np.zeros(input_image.shape[:-1], dtype="uint8")
                chromosome_img_mask[markers == m] = 1
                chromosome_img_mask = cv.morphologyEx(chromosome_img_mask, cv.MORPH_DILATE, kernel, iterations=3)
                mask_labelled = label(chromosome_img_mask)
                mask_labelled_regions = regionprops(mask_labelled)
                if len(mask_labelled_regions) > 0:
                    bbox = mask_labelled_regions[0].bbox
                    input_cr_img = input_image[bbox[0] : bbox[2], bbox[1] : bbox[3]]
                    input_cr_c0_img = input_c0[bbox[0] : bbox[2], bbox[1] : bbox[3]]
                    input_cr_c1_img = input_c1[bbox[0] : bbox[2], bbox[1] : bbox[3]]
                    input_cr_c2_img = input_c2[bbox[0] : bbox[2], bbox[1] : bbox[3]]

                    # cv.imshow('Chromosome {}'.format(str(m)), input_cr_img)
                    cv.imwrite(os.path.join(res_chrom_dir, "chromosome{}.png".format(str(m))), input_cr_img)
                    fig = plt.figure()
                    fig.suptitle("Chromosome {}".format(str(m)))
                    fig.add_subplot(2, 2, 1)
                    plt.imshow(normalize_image(input_cr_c0_img))
                    fig.add_subplot(2, 2, 2)
                    plt.imshow(normalize_image(input_cr_c1_img))
                    fig.add_subplot(2, 2, 3)
                    plt.imshow(normalize_image(input_cr_c2_img))
                    fig.add_subplot(2, 2, 4)
                    plt.imshow(input_image[bbox[0] : bbox[2], bbox[1] : bbox[3]])
                    # plt.show()
                    plt.savefig(os.path.join(res_chrom_view_dir, "all_views_chromosome{}.png".format(str(m))))
                    plt.close(fig)

                    marker_intensity_thresh = (
                        ResourcesConfiguration.getInstance().classical_marker_intensity_threshold
                    )  # 180
                    if (
                        np.max(input_cr_c1_img) >= marker_intensity_thresh
                        or np.max(input_cr_c2_img) >= marker_intensity_thresh
                    ):
                        highlighted_cr_bboxes.append(bbox)
                        highlighted_cr_numbers.append(m)

                        # @TODO. Should in addition check that there are at least two colored-dot regions
                        # (and not side from another chromosome)
                        if (
                            np.max(input_cr_c1_img) >= marker_intensity_thresh
                            and not np.max(input_cr_c2_img) >= marker_intensity_thresh
                        ) or (
                            not np.max(input_cr_c1_img) >= marker_intensity_thresh
                            and np.max(input_cr_c2_img) >= marker_intensity_thresh
                        ):
                            total_nonzero = np.count_nonzero(
                                input_cr_c1_img > marker_intensity_thresh
                            ) + np.count_nonzero(input_cr_c2_img > marker_intensity_thresh)
                            if total_nonzero > 2:
                                suspicious_cr_numbers.append(m)
            except Exception as e:
                if ResourcesConfiguration.getInstance().default_error_handling == "log":
                    logging.error("""Translocation identification failed with:\n{}""".format(e))
                    continue
                elif ResourcesConfiguration.getInstance().default_error_handling == "break":
                    raise ValueError(f"{e}")
    except Exception as e:
        raise ValueError(f"Translocation identification process failed with: {e}")
    return highlighted_cr_bboxes, highlighted_cr_numbers, suspicious_cr_numbers
