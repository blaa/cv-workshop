#!/usr/bin/env python3

"""
Computer vision workshop - empty template for participants.

Prepared for "Exatel Security Days Programming Workshop" @07.06.2019
Author: Tomasz bla Fortuna.
License: Apache
"""

import sys
from time import time

import cv2
import numpy as np


class Servo:
    """
    Servo control using PWM on Raspberry PI.
    """

    def __init__(self, servo_pin=18):
        try:
            import pigpio
            self.pi = pigpio.pi()
        except ImportError:
            print("No pigpio - simulating SERVO")
            self.pi = None

    def set(self, value):
        if self.pi is not None:
            self.pi.set_servo_pulsewidth(18, value)


servo = Servo()


class Capture:
    """
    Read data from camera and do initial, basic filtering.
    """
    def __init__(self, filename=None, camera=None, size=None):
        """
        Push filename if reading from file or camera if reading from camera. Don't use both.

        Size if you want to force capture size.
        """
        self.filename = filename
        self.camera = camera

        if camera is not None:
            self.capture = cv2.VideoCapture(camera)

            if size is not None:
                width, height = size
                self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                print("Set size to %dx%d" % (width, height))

        elif filename is not None:
            self.capture = cv2.VideoCapture(filename)
        else:
            raise Exception("You failed at thinking")

        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print("Initialized capturing device, size: %dx%d" % (self.width,
                                                             self.height))

    def frames(self):
        "Frame iterator - grab and yield frame."
        while True:
            start = time()
            (grabbed, frame) = self.capture.read()
            took = time() - start

            if frame is None:
                break

            yield frame


def main_loop(filename):
    "Loop and detect"
    if '/dev/' in filename:
        # A bit hacky, but works.
        capture = Capture(camera=filename)
    else:
        capture = Capture(filename)

    # Stats
    start = time()
    frame_no = 0

    for frame in capture.frames():
        # Let's count FPS
        frame_no += 1
        if frame_no % 10 == 0:
            took = time() - start
            print("fps: %.2f" % (frame_no/took))


        cv2.imshow("Frame", frame)
        x = cv2.waitKey(25)
        if x == ord('q'):
            break

        # Save if you want to.
        #output.write(frame)

    # Destroy windows. If any.
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main_loop(sys.argv[1])
    except KeyboardInterrupt:
        print("Exiting on keyboard interrupt at", end=' ')
