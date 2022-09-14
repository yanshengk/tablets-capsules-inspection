import my_function as mf
import json


with open("data.json", "r") as f:
    samples_data = json.load(f)

sample_cluster = dict()
sample_hsv1 = dict()
sample_hsv2 = dict()
sample_object = dict()
sample_location = dict()
sample_size = dict()
sample_area = dict()

for data in samples_data["samples"]:
    _id = data["id"]
    _cluster = data["cluster"]
    _hsv1 = data["hsv1"]
    _hsv2 = data["hsv2"]
    _object = data["object"]
    _location = data["location"]
    _size = data["size"]
    _area = data["area"]

    sample_cluster[_id] = _cluster
    sample_hsv1[_id] = _hsv1
    sample_hsv2[_id] = _hsv2
    sample_object[_id] = _object
    sample_location[_id] = _location
    sample_size[_id] = _size
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
            box_colour.append((0, 128, 255))
            flag = False
        else:
            box_colour.append((0, 255, 0))
        count += 1

    if count != sample_object[sample]:
        flag = False

        centre = []
        for b in box:
            x, y, width, height = b
            centre.append([x + (width // 2), y + (height // 2)])
        print(centre)

        presence = []
        i = 0
        for c in centre:
            for lct in sample_location[sample]:
                if lct[0] - sample_size[sample][0] // 2 < c[0] < lct[0] + sample_size[sample][0] // 2:
                    if lct[1] - sample_size[sample][1] // 2 < c[1] < lct[1] + sample_size[sample][1] // 2:
                        presence.append(i)
                i += 1
            i = 0
        print(presence)

        missing = []
        j = 0
        for _ in range(sample_object[sample]):
            if j not in presence:
                missing.append(j)
            j += 1
        print(missing)

        for m in missing:
            x = sample_location[sample][m][0] - sample_size[sample][0] // 2
            y = sample_location[sample][m][1] - sample_size[sample][1] // 2
            width = sample_size[sample][0]
            height = sample_size[sample][1]

            contours.append(None)
            box.append((x, y, width, height))
            box_colour.append((0, 0, 255))

    img_result = mf.draw_objects(img_roi, contours, box, box_colour, path=path)

    return count, flag, img_result
