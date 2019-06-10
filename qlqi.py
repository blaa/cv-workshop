#!/usr/bin/env python3

"""
Computer vision workshop template - after filling!

NOTE: This wasn't a workshop on a correct Python program composition.

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
    Read data from camera or from a file.
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
            (grabbed, frame) = self.capture.read()

            if frame is None:
                break

            yield frame


def detect_kulka(frame, diff):
    """
    Ball (kulka) detection algorithm.

    Args:
      diff: difference from the averaged background.
      frame: current color frame.
    """
    # Erosion/dilatation morphology filtering:
    # Erosion removes small artefacts, but "thins" our difference.
    # Dilatation brings "thickness" back, but not on removed artefacts.
    kernel = np.ones((5,5),np.uint8)
    diff = cv2.erode(diff, kernel, iterations=1)
    diff = cv2.dilate(diff, kernel, iterations=3)

    # Treshold "diff" to get a "mask" with our ball.
    status, mask = cv2.threshold(diff, 20, 255,
                                 cv2.THRESH_BINARY)

    # Convert BGR frame to HSV to get "Hue".
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Calculate histogram of the Hue (channel 0 of HSV)
    h_hist = cv2.calcHist([hsv], channels=[0],
                          mask=mask, histSize=[6],
                          ranges=[0, 179])

    # Heuristic to differentiate red and non-red color (edges of histogram vs center)
    edge = h_hist[0] + h_hist[-1]
    center = sum(h_hist) - edge

    if abs(edge - center) < 200:
        # Difference not big enough.
        print("INVALID KULKA", edge, center)
        return None
    if edge > 2000:
        # Red ball found.
        print("RED KULKA", edge, center)
        servo.set(1800)
    else:
        # Non-red ball found.
        print("NON-RED KULKA", edge, center)
        servo.set(1500)


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

    # "Motion detected" frame counter.
    det_cnt = 0

    # Previous frame (for motion detection)
    prev_frame = None

    # Averaged background (for ball-mask calculation)
    background = None

    for frame in capture.frames():
        # Let's count FPS
        frame_no += 1
        if frame_no % 10 == 0:
            took = time() - start
            print("fps: %.2f" % (frame_no/took))

        # Get grey frame to speed up "motion detection" on rPI. 1 channel needs
        # to be calculated later, not 3 of them.
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_frame is None:
            # Initialize previous frame and background.
            prev_frame = grey
            background = grey
            continue

        # absdiff calculates absolute difference on each pixel without problems
        # of saturation of overflowing int8 (0-255) ranges.
        diff = cv2.absdiff(grey, prev_frame)

        # Saturated subtraction - when subtracting 30 from 20 we get 0. So we
        # remove all the low-value noise from the diff.
        diff = cv2.subtract(diff, 30)

        # Calculate total brightness of all not-filtered pixels on the
        # difference image.
        diff_total = diff.sum()

        if diff_total > 1000:
            # Difference exceeds some threshold - movement detected, count frames.
            det_cnt += 1
        else:
            # No movement - reset movement counters.
            det_cnt = 0
            # And "running average" our background.
            background = cv2.addWeighted(background, 0.9, grey, 0.1, 0)

        if det_cnt == 3:
            # On Xth moving frame we detect the ball color.

            # Calculate difference from the background - does better job then
            # difference on the two consecutive frames, because we get the
            # difference only where the ball IS and not - where is was on the
            # previous frame.
            diff = cv2.absdiff(background, grey)

            # That's how you can display some frame:
            #cv2.imshow("NEW DIFF", diff)

            # Call ball detection algorithm.
            detect_kulka(frame, diff)

        prev_frame = grey

        # Remove imshows if running on rpi.
        cv2.imshow("Frame", frame)

        x = cv2.waitKey(25)
        if x == ord('q'):
            break

        # You can easily save some interesting frames to the disc.
        #output.write(frame)

    # Destroy windows. If any.
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main_loop(sys.argv[1])
    except KeyboardInterrupt:
        print("Exiting on keyboard interrupt at", end=' ')
