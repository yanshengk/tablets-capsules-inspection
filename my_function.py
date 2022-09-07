import os
import imutils
import cv2
import numpy as np


def print_message(message, category="    "):
    os.system("date")
    print(f"\033[1m[{category}]\033[0m  {message}")


def make_directory(parent, child):
    path = os.path.join(parent, child)

    if os.path.exists(path):
        print_message(f"\"{child}\" directory already exists", "INFO")
    else:
        os.mkdir(path)
        print_message(f"Successfully created directory \"{child}\"", "INFO")


def resize_image(image, factor):
    width = int(image.shape[1] * factor)
    result = imutils.resize(image, width=width)

    return result


def stack_images(scale, img_array):
    rows = len(img_array)
    cols = len(img_array[0])
    rows_available = isinstance(img_array[0], list)
    width = img_array[0][0].shape[1]
    height = img_array[0][0].shape[0]
    if rows_available:
        for x in range(0, rows):
            for y in range(0, cols):
                if img_array[x][y].shape[:2] == img_array[0][0].shape[:2]:
                    img_array[x][y] = cv2.resize(img_array[x][y], (0, 0), None, scale, scale)
                else:
                    img_array[x][y] = cv2.resize(img_array[x][y], (img_array[0][0].shape[1], img_array[0][0].shape[0]),
                                                 None, scale, scale)
                if len(img_array[x][y].shape) == 2:
                    img_array[x][y] = cv2.cvtColor(img_array[x][y], cv2.COLOR_GRAY2BGR)
        image_blank = np.zeros((height, width, 3), np.uint8)
        hor = [image_blank] * rows
        # hor_con = [image_blank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(img_array[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if img_array[x].shape[:2] == img_array[0].shape[:2]:
                img_array[x] = cv2.resize(img_array[x], (0, 0), None, scale, scale)
            else:
                img_array[x] = cv2.resize(img_array[x], (img_array[0].shape[1], img_array[0].shape[0]), None, scale,
                                          scale)
            if len(img_array[x].shape) == 2:
                img_array[x] = cv2.cvtColor(img_array[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(img_array)
        ver = hor
    return ver


def hsv_boundary(target):
    h = target[0]
    s = target[1]
    v = target[2]

    h_min = h - target[3]
    h_max = h + target[3]

    s_min = s - target[3]
    s_max = s + target[3]

    v_min = v - target[3]
    v_max = v + target[3]

    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])

    return lower, upper


def get_roi(image, path=None):
    hsv = bgr_to_hsv(image)
    blur = blur_image(hsv)
    mask = mask_image(blur, [95, 55, 60, 50])
    canny = canny_detection(mask)
    dilate = dilate_image(canny)

    x, y, width, height = (550, 930, 3400, 2200)

    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000000:
            perimeter = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
            x, y, width, height = cv2.boundingRect(approx)

    roi = image[y:y + height, x:x + width]
    if path is not None:
        cv2.imwrite(path + "_ROI.jpg", roi)

    return roi


def bgr_to_hsv(image, path=None):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    if path is not None:
        cv2.imwrite(path + "_HSV.jpg", hsv)

    return hsv


def kmeans_clustering(image, k, path=None):
    # reshape the image to a 2D array of pixels and 3 color values (RGB)
    two_d_image = image.reshape((-1, 3))
    # convert to float
    two_d_image = np.float32(two_d_image)
    # print(two_d_image.shape)

    # define stopping criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0)
    # apply kmeans
    _, labels, centers = cv2.kmeans(two_d_image, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    # print(centers)

    # convert back to 8 bit values
    centers = np.uint8(centers)

    # convert all pixels to the color of the centroids
    segmented = centers[labels.flatten()]

    # reshape back to the original image dimension
    segmented = segmented.reshape(image.shape)
    if path is not None:
        cv2.imwrite(path + "_SEGMENTED.jpg", segmented)

    return segmented


def blur_image(image, path=None):
    blur = cv2.GaussianBlur(image, (35, 35), 2.5)
    if path is not None:
        cv2.imwrite(path + "_BLUR.jpg", blur)

    return blur


def mask_image(image, target1, target2=None, path=None):
    if target2 is None:
        lower, upper = hsv_boundary(target1)

        mask = cv2.inRange(image, lower, upper)
    else:
        lower1, upper1 = hsv_boundary(target1)
        lower2, upper2 = hsv_boundary(target2)

        img_mask1 = cv2.inRange(image, lower1, upper1)
        img_mask2 = cv2.inRange(image, lower2, upper2)
        mask = img_mask1 | img_mask2
    if path is not None:
        cv2.imwrite(path + "_MASK.jpg", mask)

    return mask


def canny_detection(image, path=None):
    canny = cv2.Canny(image, 100, 300)
    if path is not None:
        cv2.imwrite(path + "_CANNY.jpg", canny)

    return canny


def dilate_image(image, path=None):
    kernel = np.ones((3, 3), np.uint8)
    dilate = cv2.dilate(image, kernel, iterations=1)
    if path is not None:
        cv2.imwrite(path + "_DILATE.jpg", dilate)

    return dilate


def get_objects(image):
    all_contours = []
    all_area = []
    all_box = []

    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 10000:
            all_contours.append(cnt)
            all_area.append(area)

            perimeter = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
            x, y, width, height = cv2.boundingRect(approx)
            all_box.append((x, y, width, height))

    return all_contours, all_area, all_box


def draw_objects(image, sample, contours, area, box, path=None):
    index = 0
    flag = True
    if sample == 1:
        objects = 10
        area_threshold = 75000
    elif sample == 2:
        objects = 12
        area_threshold = 45000
    else:
        objects = 10
        area_threshold = 60000
    result = image.copy()

    for a in area:
        cv2.drawContours(result, contours[index], -1, (255, 0, 0), 3)

        x, y, width, height = box[index]

        if a < area_threshold:
            bounding_box_colour = (0, 0, 255)
            flag = False
        else:
            bounding_box_colour = (0, 255, 0)
        cv2.rectangle(result, (x, y), (x + width, y + height), bounding_box_colour, 3)

        index += 1
    if path is not None:
        cv2.imwrite(path + "_RESULT.jpg", result)

    if index != objects:
        flag = False

    return result, index, flag
