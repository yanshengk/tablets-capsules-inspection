import cv2
import numpy as np


def hsv_boundary(target, sensitivity):
    h = target[0]
    s = target[1]
    v = target[2]

    h_min = h - sensitivity
    h_max = h + sensitivity

    s_min = s - sensitivity
    s_max = s + sensitivity

    v_min = v - sensitivity
    v_max = v + sensitivity

    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])

    return lower, upper


def get_roi(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    blur = cv2.GaussianBlur(hsv, (35, 35), 2.5)
    lower, upper = hsv_boundary([95, 55, 60], 50)
    mask = cv2.inRange(blur, lower, upper)
    canny = cv2.Canny(mask, 100, 300)
    kernel = np.ones((3, 3), np.uint8)
    dilate = cv2.dilate(canny, kernel, iterations=1)

    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > 1000000:
            perimeter = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)

            return cv2.boundingRect(approx)

    return 550, 930, 3400, 2200


def process_image(image, path, sample):
    # Crop ROI from image
    # img_roi = image[930:3130, 550:3950]  # (3400, 2200)
    x, y, width, height = get_roi(image)
    img_roi = image[y:y + height, x:x + width]
    cv2.imwrite(path + "_ROI.jpg", img_roi)

    # Convert image to HSV colour space
    img_hsv = cv2.cvtColor(img_roi, cv2.COLOR_BGR2HSV)
    cv2.imwrite(path + "_HSV.jpg", img_hsv)

    # reshape the image to a 2D array of pixels and 3 color values (RGB)
    img_reshape = img_hsv.reshape((-1, 3))
    # convert to float
    img_reshape = np.float32(img_reshape)
    # print(img_reshape.shape)

    # define stopping criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0)
    # number of clusters (k)
    if sample == 1 or sample == 2:
        k = 3
    else:
        k = 6
    # apply kmeans
    _, labels, centers = cv2.kmeans(img_reshape, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    # print(centers)

    # convert back to 8 bit values
    centers = np.uint8(centers)

    # convert all pixels to the color of the centroids
    img_segmented = centers[labels.flatten()]

    # reshape back to the original image dimension
    img_segmented = img_segmented.reshape(img_hsv.shape)
    cv2.imwrite(path + "_SEGMENTED.jpg", img_segmented)

    # Blur image
    img_blur = cv2.GaussianBlur(img_segmented, (35, 35), 2.5)
    cv2.imwrite(path + "_BLUR.jpg", img_blur)

    # Set HSV target boundary and mask target
    if sample == 1:  # WHITE TABLETS
        target = [52, 9, 224, 10]
        lower, upper = hsv_boundary(target[0:3], target[3])

        img_mask = cv2.inRange(img_blur, lower, upper)
    elif sample == 2:  # PINK TABLETS
        target = [165, 29, 208, 20]
        lower, upper = hsv_boundary(target[0:3], target[3])

        img_mask = cv2.inRange(img_blur, lower, upper)
    else:  # YELLOW-GREEN CAPSULES
        target1 = [75, 88, 166, 20]
        target2 = [28, 121, 192, 20]
        lower1, upper1 = hsv_boundary(target1[0:3], target1[3])
        lower2, upper2 = hsv_boundary(target2[0:3], target2[3])

        img_mask1 = cv2.inRange(img_blur, lower1, upper1)
        img_mask2 = cv2.inRange(img_blur, lower2, upper2)
        img_mask = img_mask1 | img_mask2
    cv2.imwrite(path + "_MASK.jpg", img_mask)

    # Perform canny edge detection
    img_canny = cv2.Canny(img_mask, 100, 300)
    cv2.imwrite(path + "_CANNY.jpg", img_canny)

    # Dilate image
    kernel = np.ones((3, 3), np.uint8)
    img_dilate = cv2.dilate(img_canny, kernel, iterations=1)
    cv2.imwrite(path + "_DILATE.jpg", img_dilate)

    # Get contours
    obj_count = 0
    condition = True
    if sample == 1:
        actual_count = 10
        area_threshold = 75000
    elif sample == 2:
        actual_count = 12
        area_threshold = 45000
    else:
        actual_count = 10
        area_threshold = 60000
    img_contours = img_roi.copy()
    contours, hierarchy = cv2.findContours(img_dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 10000:
            # cv2.drawContours(img_contours, cnt, -1, (255, 0, 0), 3)
            perimeter = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
            x, y, width, height = cv2.boundingRect(approx)
            if area < area_threshold:
                bounding_box_colour = (0, 0, 255)
                condition = False
            else:
                bounding_box_colour = (0, 255, 0)
            cv2.rectangle(img_contours, (x, y), (x + width, y + height), bounding_box_colour, 3)
            # cv2.putText(img_contours, str(area), (x + (width // 2) - 30, y + (height // 2) + 5),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            obj_count += 1
    cv2.imwrite(path + "_CONTOURS.jpg", img_contours)

    if obj_count != actual_count:
        condition = False

    return img_contours, obj_count, condition
