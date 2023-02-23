import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def read_image_label(path_to_img, path_to_txt, txt_row_obj=0, normilize=True):

    """Read image and corresponding label (.txt) file with coordinates of nail
    polygon

    Args:
        path_to_img: str, path to image
        path_to_txt: str, path to .txt file
        txt_row_obj: int, row number in .txt file with coordinates of nail polygon
        normilize: bool, if True, normilize coordinates of nail polygon

    Returns:
        image: numpy array, image
        polygon_nail: numpy array, coordinates of nail polygon
        obj_class: str, nail class
    """

    # read image
    image = cv2.imread(path_to_img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # read corresponding .txt file
    with open(path_to_txt, "r") as f:
        txt_row = f.readlines()[txt_row_obj].split()
        obj_class = txt_row[0]
        coords = txt_row[1:]
        polygon_nail = np.array(
            [[eval(x), eval(y)] for x, y in zip(coords[0::2], coords[1::2])]
        )  # convert list of coordinates to numpy massive

    # Convert normilized coordinates of polygon_nail to pixel coordinates
    if normilize:
        img_h, img_w = image.shape[:2]
        polygon_nail[:, 0] = polygon_nail[:, 0] * img_w
        polygon_nail[:, 1] = polygon_nail[:, 1] * img_h

    return image, polygon_nail.astype(int), obj_class


def get_image_mask(img, polygon_nail, display_mask=False):

    """Create mask on the image, showing the segmented nail

    Args:
        img: numpy array, image
        polygon_nail: numpy array, coordinates of nail polygon
        display_mask: bool, if True, display image with mask

    Returns:
        mask: numpy array, mask of nail on the image
    """

    # Create zero array for mask
    mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

    # Draw polygon_nail on the image and mask
    cv2.fillPoly(mask, pts=[polygon_nail], color=(255, 255, 255))

    if display_mask:
        fig = plt.figure(figsize=(22, 18))
        axes = fig.subplots(nrows=1, ncols=2)
        axes[0].imshow(img)
        axes[1].imshow(mask, cmap="Greys_r")
        axes[0].set_title("Original image with mask")
        axes[1].set_title("Mask")
        plt.show()
        masked = cv2.bitwise_and(img, img, mask=mask)
        fig = plt.figure(figsize=(22, 18))
        axes = fig.subplots(nrows=1, ncols=1)
        axes.imshow(masked)

    return mask


def crop_image_label(
    image, polygon_nail, square=True, max_extra_pad_prop=0.2, obj_class=0
):

    """Crop image around nail polygon with specified proportion of image area
    (padding) shown around the nail

    Args:
        image: numpy array, image
        polygon_nail: numpy array, coordinates of nail polygon
        square: bool, if True, adds image padding around the nail so that image
        is square (only if img around the nail is big enough)
        max_extra_pad_prop: float, max proportion of extra image padding to add
        around the nail
        obj_class: str, object class in the image

    Returns:
        image: numpy array, cropped image
        polygon_nail: numpy array, cropped coordinates of nail polygon
        obj_class: str, nail class
    """

    img_h, img_w = image.shape[:2]

    # min/max of nail mask
    min_x = min(polygon_nail[:, 0])
    min_y = min(polygon_nail[:, 1])
    max_x = max(polygon_nail[:, 0])
    max_y = max(polygon_nail[:, 1])

    if square == True:

        # if height is greater than width
        if (max_y - min_y) > (max_x - min_x):

            # calculate the padding on width-dim needed to make the image square
            min_x_pad = min_x - int(
                np.round(((max_y - min_y) - (max_x - min_x)) / 2)
            )  # pad left
            max_x_pad = max_x + int(
                np.round(((max_y - min_y) - (max_x - min_x)) / 2)
            )  # pad right

            # apply padding if it doesn't exceed the original image width
            if min_x_pad > 0 and max_x_pad < img_w:
                min_x = min_x_pad
                max_x = max_x_pad

        # if width is greater than height
        elif (max_y - min_y) < (max_x - min_x):

            # calculate the padding on height-dim needed to make the image square
            min_y_pad = min_y - int(
                np.round(((max_x - min_x) - (max_y - min_y)) / 2)
            )  # pad top
            max_y_pad = max_y + int(
                np.round(((max_x - min_x) - (max_y - min_y)) / 2)
            )  # pad bottom

            # apply padding if it doesn't exceed the original image height
            if min_y_pad > 0 and max_y_pad < img_h:
                min_y = min_y_pad
                max_y = max_y_pad

    if max_extra_pad_prop > 0:

        # calculate distance from nail mask to edge of image
        dist_right = img_w - max_x
        dist_left = min_x
        dist_top = min_y
        dist_bottom = img_h - max_y

        mask_width = max_x - min_x  # TL added this
        mask_height = max_y - min_y  # TL added this
        smallest_dim = min([mask_width, mask_height])  # TL added this

        # get lowest distance
        min_dist = min([dist_right, dist_left, dist_top, dist_bottom])

        # if extra padding is greater extra pad will be set to the lowest distance
        if int(max_extra_pad_prop * smallest_dim) > min_dist:
            max_extra_pad_prop = min_dist / smallest_dim

        min_x, min_y, max_x, max_y = (
            min_x - int(max_extra_pad_prop * smallest_dim),
            min_y - int(max_extra_pad_prop * smallest_dim),
            max_x + int(max_extra_pad_prop * smallest_dim),
            max_y + int(max_extra_pad_prop * smallest_dim),
        )

    image_cropped = image[min_y:max_y, min_x:max_x, :]
    polygon_nail_cropped = polygon_nail - np.array([min_x, min_y])

    # convert poligon back to normilized coordinates (now of the new cropped image dims)
    polygon_nail_cropped_norm = polygon_nail_cropped / np.array(
        [image_cropped.shape[1], image_cropped.shape[0]]
    )

    # convert polygon to list of coordinates in original format
    polygon_nail_cropped_norm = polygon_nail_cropped_norm.reshape(-1).tolist()
    polygon_nail_cropped_norm = [obj_class] + polygon_nail_cropped_norm

    return image_cropped, polygon_nail_cropped, polygon_nail_cropped_norm


def save_image_label(
    image_cropped,
    polygon_nail_cropped_norm,
    imgs_cropped_path,
    labels_cropped_path,
    img_file,
    txt_file,
):

    """Save cropped image and nail polygon coordinates to file"""

    # save cropped image
    cv2.imwrite(
        os.path.join(imgs_cropped_path, img_file),
        cv2.cvtColor(image_cropped, cv2.COLOR_BGR2RGB),
    )

    # save coordinates to txt file
    with open(os.path.join(labels_cropped_path, txt_file), "w") as f:
        for item in polygon_nail_cropped_norm:
            f.write("%s " % item)
