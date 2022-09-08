import my_function as mf
import json


with open("data.json", "r") as f:
    samples_data = json.load(f)

sample_cluster = dict()
sample_hsv1 = dict()
sample_hsv2 = dict()
sample_object = dict()
sample_area = dict()

for data in samples_data["samples"]:
    _id = data["id"]
    _cluster = data["cluster"]
    _hsv1 = data["hsv1"]
    _hsv2 = data["hsv2"]
    _object = data["object"]
    _area = data["area"]

    sample_cluster[_id] = _cluster
    sample_hsv1[_id] = _hsv1
    sample_hsv2[_id] = _hsv2
    sample_object[_id] = _object
    sample_area[_id] = _area


def process_image(image, path, sample):
    img_roi = mf.get_roi(image, path=path)

    img_hsv = mf.bgr_to_hsv(img_roi, path=path)

    img_segmented, colours = mf.kmeans_clustering(img_hsv, sample_cluster[sample], path=path)
    print(colours)

    img_blur = mf.blur_image(img_segmented, path=path)

    img_mask = mf.mask_image(img_blur, sample_hsv1[sample], sample_hsv2[sample], path=path)

    img_canny = mf.canny_detection(img_mask, path=path)

    img_dilate = mf.dilate_image(img_canny, path=path)

    contours, area, box = mf.get_objects(img_dilate)

    flag = True
    count = 0
    box_colour = []
    for a in area:
        if a < sample_area[sample]:
            box_colour.append((0, 0, 255))
            flag = False
        else:
            box_colour.append((0, 255, 0))
        count += 1

    if count != sample_object[sample]:
        flag = False

    img_result = mf.draw_objects(img_roi, contours, box, box_colour, path=path)

    return count, flag, img_result
