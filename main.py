import RPi.GPIO as GPIO
import os
from datetime import datetime
import cv2
# import numpy as np
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
GPIO.output(LED, GPIO.HIGH)

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


def capture_image(ev, preview=False, duration=1500):
    default = f"libcamera-still -t {duration} --rotation 180 --autofocus --ev {ev} -q 100 -o image.jpg"

    if not preview:
        my_cmd = default + " --nopreview"
    else:
        my_cmd = default

    GPIO.output(LED, GPIO.LOW)

    os.system(my_cmd)

    GPIO.output(LED, GPIO.HIGH)


def main(sample):
    # Start conveyor until carriage is detected
    while True:
        if GPIO.input(INFRARED):
            motor(75, 1)
        else:
            motor(0, 0)
            break

    # Capture image and store in a new directory
    capture_image(0)
    current_timestamp = str(datetime.now()).replace(" ", "_")
    mf.make_directory("Images/", current_timestamp)
    os.rename(f"image.jpg", f"Images/{current_timestamp}/{current_timestamp}.jpg")
    mf.print_message("Successfully captured and saved", "INFO")

    # Define image path
    path = f"Images/{current_timestamp}/{current_timestamp}"

    # Read image
    img = cv2.imread(path + ".jpg")

    mf.print_message("Processing...", "INFO")

    # Process image
    count, condition, result = mp.process_image(img, path, sample)

    # Show result
    cv2.imshow("Contours", mf.resize_image(result, 0.40))
    mf.print_message(f"{count} object(s) detected", "INFO")
    if condition:
        mf.print_message("PASSED", "INFO")
    else:
        mf.print_message("FAILED", "INFO")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    while True:
        if GPIO.input(INFRARED):
            motor(0, 0)
            break
        else:
            motor(75, 1)


if __name__ == "__main__":
    main(3)

    GPIO.cleanup()
