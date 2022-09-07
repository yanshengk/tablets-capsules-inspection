import my_function as mf


def process_image(image, path, sample):
    img_roi = mf.get_roi(image, path=path)

    img_hsv = mf.bgr_to_hsv(img_roi, path=path)

    if sample == 1 or sample == 2:
        k = 3
    else:
        k = 6
    img_segmented = mf.kmeans_clustering(img_hsv, k, path=path)

    img_blur = mf.blur_image(img_segmented, path=path)

    if sample == 1 or sample == 2:
        if sample == 1:  # WHITE TABLETS
            target = [52, 9, 224, 10]
        else:  # PINK TABLETS
            target = [165, 29, 208, 20]
        img_mask = mf.mask_image(img_blur, target, path=path)
    else:
        # YELLOW-GREEN CAPSULES
        target1 = [75, 88, 166, 20]
        target2 = [28, 121, 192, 20]
        img_mask = mf.mask_image(img_blur, target1, target2, path=path)

    img_canny = mf.canny_detection(img_mask, path=path)

    img_dilate = mf.dilate_image(img_canny, path=path)

    contours, area, box = mf.get_objects(img_dilate)

    img_result, count, flag = mf.draw_objects(img_roi, sample, contours, area, box, path=path)

    return img_result, count, flag
