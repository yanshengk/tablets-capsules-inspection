import RPi.GPIO as GPIO
import os
from datetime import datetime
import cv2
import imutils
import numpy as np
# from matplotlib import pyplot as plt
import my_function as mf
import my_process as mp

MOTOR_1_A = 22
MOTOR_1_B = 27
MOTOR_1_EN = 18
INFRARED = 4
LED = 23  # Relay 1
UNUSED = 24  # Relay 2

GPIO.setmode(GPIO.BCM)
GPIO.setup(MOTOR_1_A, GPIO.OUT)
GPIO.setup(MOTOR_1_B, GPIO.OUT)
GPIO.setup(MOTOR_1_EN, GPIO.OUT)
GPIO.setup(INFRARED, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(LED, GPIO.OUT)

pwm = GPIO.PWM(MOTOR_1_EN, 50)


def motor(speed, direction=1):
    if direction == 1:
        GPIO.output(MOTOR_1_A, GPIO.HIGH)
        GPIO.output(MOTOR_1_B, GPIO.LOW)
        pwm.start(speed)
    elif direction == -1:
        GPIO.output(MOTOR_1_A, GPIO.LOW)
        GPIO.output(MOTOR_1_B, GPIO.HIGH)
        pwm.start(speed)
    elif direction == 0:
        GPIO.output(MOTOR_1_A, GPIO.LOW)
        GPIO.output(MOTOR_1_B, GPIO.LOW)
        pwm.stop()


def img_capture(ev, preview=False, duration=1500):
    current_timestamp = str(datetime.now()).replace(" ", "_")

    mf.make_directory("", "Images")
    mf.make_directory("Images/", current_timestamp)

    default = f"libcamera-still -t {duration} --rotation 180 --autofocus --ev {ev} -o {current_timestamp}.jpg"

    if not preview:
        my_cmd = default + " --nopreview"
    else:
        my_cmd = default

    GPIO.output(LED, GPIO.LOW)

    os.system(my_cmd)

    GPIO.output(LED, GPIO.HIGH)

    os.rename(f"{current_timestamp}.jpg", f"Images/{current_timestamp}/{current_timestamp}.jpg")

    mf.print_message("Successfully captured and saved", "INFO")

    return current_timestamp


def img_resize(image, factor):
    width = int(image.shape[1] * factor)
    image_resized = imutils.resize(image, width=width)

    return image_resized


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


def main():
    # Start conveyor until carriage is detected
    while True:
        if GPIO.input(INFRARED):
            motor(75, 1)
        else:
            motor(0, 0)
            break

    # Capture image and create a new directory
    img_directory = img_capture(0)

    # Define image path
    path = f"Images/{img_directory}/{img_directory}"

    # Read image
    img = cv2.imread(path + ".jpg")

    mf.print_message("Processing...", "INFO")

    # Process image
    result, count, condition = mp.process_image(img, path, 3)  # 1-WhiteTab, 2-PinkTab, 3-YellowGreenCap

    # Show output
    cv2.imshow("Contours", img_resize(result, 0.40))
    mf.print_message(f"{count} object(s) detected", "INFO")
    if condition:
        mf.print_message("PASSED", "INFO")
    else:
        mf.print_message("FAILED", "INFO")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Show output
    # imgBlank = np.zeros_like(imgROI)
    # imgStack = stackImages(0.2, ([imgResult, imgCanny, imgDilate], [imgContours, imgBlank, imgBlank]))

    # while True:
    #     if GPIO.input(INFRARED):
    #         motor(0, 0)
    #         break
    #     else:
    #         motor(75, 1)


if __name__ == "__main__":
    main()

    GPIO.cleanup()
