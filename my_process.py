import cv2
import numpy as np
import my_function as mf


def get_contours(image, image_display):
    obj_count = 0

    image_copy = image_display.copy()

    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > 10000:
            cv2.drawContours(image_copy, cnt, -1, (255, 0, 0), 3)

            perimeter = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
            x, y, width, height = cv2.boundingRect(approx)

            if perimeter < 1400:
                cv2.rectangle(image_copy, (x, y), (x + width, y + height), (0, 255, 0), 2)
            else:
                cv2.rectangle(image_copy, (x, y), (x + width, y + height), (0, 0, 255), 2)
            cv2.putText(image_copy, str(perimeter), (x + (width // 2) - 30, y + (height // 2) + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            obj_count += 1

    mf.print_message(f"{obj_count} object(s) detected", "INFO")

    return image_copy


def process_tablets(image, path, target):
    # Crop ROI from image
    img_roi = image[930:3130, 550:3950]  # (3400, 2200)
    cv2.imwrite(path + "_ROI.jpg", img_roi)

    # Blur image
    img_blur = cv2.GaussianBlur(img_roi, (35, 35), 2.5)
    cv2.imwrite(path + "_BLUR.jpg", img_blur)

    # Convert image to HSV colour space
    img_hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)
    cv2.imwrite(path + "_HSV.jpg", img_hsv)

    # HSV target boundary
    sensitivity = target[3]  # 20

    h_target = target[0]  # 40
    s_target = target[1]  # 9
    v_target = target[2]  # 236

    h_min = h_target - sensitivity
    h_max = h_target + sensitivity

    s_min = s_target - sensitivity
    s_max = s_target + sensitivity

    v_min = v_target - sensitivity
    v_max = v_target + sensitivity

    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])

    # Mask target
    img_mask = cv2.inRange(img_hsv, lower, upper)
    cv2.imwrite(path + "_MASK.jpg", img_mask)

    img_result = cv2.bitwise_and(img_blur, img_blur, mask=img_mask)
    cv2.imwrite(path + "_RESULT.jpg", img_result)

    # Perform canny edge detection
    img_canny = cv2.Canny(img_result, 100, 300)
    cv2.imwrite(path + "_CANNY.jpg", img_canny)

    # Dilate image
    kernal = np.ones((3, 3))
    img_dilate = cv2.dilate(img_canny, kernal, iterations=1)
    cv2.imwrite(path + "_DILATE.jpg", img_dilate)

    # Get contours
    img_contours = get_contours(img_dilate, img_roi)
    cv2.imwrite(path + "_CONTOURS.jpg", img_contours)
