#! /usr/bin/python3
"""
Interferometer/PSF simulator
Created by: Jack Hickish
Minor Modifications by: Griffin Foster
Modified 14/06/16 -- Added command entry with curses. Added pause and "persistence" mode
Modified Jan 16th 2019 -- CJ, Edgar, + Adam edit. Plot 3 images in one window.
"""

import numpy as np
import time
import sys
import optparse
import curses
import os
import cv2 as cv
import scipy.ndimage


def fast_conv(image, uv_tracks):
    # Shift zero-frequency component to match centered UV tracks
    uv_tracks_shifted = np.fft.ifftshift(uv_tracks)
    uv_image = np.fft.fft2(image)
    sample = uv_image * uv_tracks_shifted
    dirty = np.abs(np.fft.ifft2(sample))
    s = np.max(dirty)
    if s > 0:
        dirty = 255.0 * (1.0 / s) * dirty
    return dirty.astype(np.uint8)


class Mutable(object):
    def __init__(self, v):
        self.v = v

    def set(self, v):
        self.v = v

    def get(self):
        return self.v

#

def find_circles(frame):
    # 1. Convert to grayscale and apply blur
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (7, 7), 0)

    # 2. Adaptive thresholding
    thresh = cv.adaptiveThreshold(
        blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 21, 10
    )

    # 3. Morphological cleaning
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
    thresh = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)

    # 4. Find all outlines (contours)
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    circles = []

    circ_threshold = 0.5
    for cnt in contours:
        area = cv.contourArea(cnt)
        if not (100 < area < 400):
            continue

        x, y, w, h = cv.boundingRect(cnt)
        aspect_ratio = float(w) / h
        if not (0.75 <= aspect_ratio <= 1.35):
            continue

        perimeter = cv.arcLength(cnt, True)
        if perimeter == 0:
            continue

        circularity = (4 * np.pi * area) / (perimeter ** 2)
        if circularity >circ_threshold:
            (cx, cy), radius = cv.minEnclosingCircle(cnt)
            center = (int(cx), int(cy))
            radius = int(radius)
            cv.circle(frame, center, radius, (0, 255, 0), 2)
            cv.circle(frame, center, 2, (0, 0, 255), -1)
            circles.append((center, radius))
    return circles

def main(stdscr, opts, args):
    CAMERA_DEVICE_INDEX = opts.camera
    cam = cv.VideoCapture(CAMERA_DEVICE_INDEX)

    if not cam.isOpened():
        raise RuntimeError(f"Could not open camera device {CAMERA_DEVICE_INDEX}")

    if opts.input is None:
        infile = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'astro_test_image.jpg')
        target_image = cv.imread(infile)
    else:
        target_image = cv.imread(opts.input)

    if target_image is None:
        raise FileNotFoundError("Could not load target input image.")

    ysize, xsize = target_image.shape[:2]
    aspect = xsize / ysize
    target_img_grey = cv.cvtColor(target_image, cv.COLOR_BGR2GRAY)

    brightness_threshold = 100
    target_img_grey[target_img_grey < brightness_threshold] = 0

    RESCALE_FACTOR = opts.res
    persistence = False
    window = False
    gaussian = True

    stdscr.nodelay(1)
    stdscr.clear()
    stdscr.addstr('Controls: Quit (q), Pause (p), Toggle Persistence (r)\n')

    # Read initial frame to determine crop dimensions
    rv, layout_img = cam.read()
    if not rv or layout_img is None:
        raise RuntimeError("Failed to read initial frame from camera.")

    yc, xc = layout_img.shape[:2]
    ac = xc / yc
    if ac > aspect:
        xc = int(np.around(aspect * yc))
    elif ac < aspect:
        yc = int(np.around(xc / aspect))

    hour_angle = Mutable(0)
    uvwindowname = "UV tracks"
    cv.namedWindow(uvwindowname, cv.WINDOW_NORMAL)
    cv.resizeWindow(uvwindowname, 600, 600)
    trackbar_name = "Hour angle"
    cv.createTrackbar(trackbar_name, uvwindowname, 0, 12, hour_angle.set)

    while True:
        rv, frame = cam.read()
        if not rv or frame is None:
            continue

        # FIXME: The old code squished things to match, which I didn't like; this crops instead and I think I like it even less.
        # Crop frame to match target image aspect ratio
        frame = frame[:yc, :xc, :].copy()

        circles = find_circles(frame)
        station_locations = [(cy, cx) for (cx, cy), _ in circles]

        # Create UV tracks for a given hour angle
        hour_angle_rad = 2 * np.pi * hour_angle.get() / 24.0
        uv_baselines = np.zeros((ysize // RESCALE_FACTOR, xsize // RESCALE_FACTOR), dtype=float)
        Y, X = uv_baselines.shape

        for i in range(len(station_locations)):
            for j in range(0, i):
                y1, x1 = station_locations[i]
                y2, x2 = station_locations[j]
                dx = int((x2 - x1) // RESCALE_FACTOR)
                dy = int((y2 - y1) // RESCALE_FACTOR)
                ha0 = np.arctan2(dx, dy)
                r = np.sqrt(dy ** 2 + dx ** 2)

                angles = np.arange(0, hour_angle_rad + 0.005, 0.005)
                try:
                    y_pos = np.clip((Y + dy) // 2, 0, Y - 1)
                    x_pos = np.clip((X + dx) // 2, 0, X - 1)
                    y_neg = np.clip((Y - dy) // 2, 0, Y - 1)
                    x_neg = np.clip((X - dx) // 2, 0, X - 1)

                    uv_baselines[y_pos, x_pos] = 1
                    uv_baselines[y_neg, x_neg] = 1

                    for ha in angles:
                        alpha = ha0 + ha
                        ay = r * np.cos(alpha)
                        ax = r * np.sin(alpha)

                        cx1 = int(np.clip(round(X + ax) // 2, 1, X - 2))
                        cy1 = int(np.clip(round(Y + ay) // 2, 1, Y - 2))
                        uv_baselines[cy1 - 1:cy1 + 1, cx1 - 1:cx1 + 1] = 1

                        cx2 = int(np.clip(round(X - ax) // 2, 1, X - 2))
                        cy2 = int(np.clip(round(Y - ay) // 2, 1, Y - 2))
                        uv_baselines[cy2 - 1:cy2 + 1, cx2 - 1:cx2 + 1] = 1
                except Exception as e:
                    stdscr.addstr(f"UV Exception: {e}\n")

        big_uv = np.zeros(target_img_grey.shape, dtype=float)
        big_uv0 = scipy.ndimage.zoom(uv_baselines, opts.res, order=0)
        ny0 = min(big_uv.shape[0], big_uv0.shape[0])
        nx0 = min(big_uv.shape[1], big_uv0.shape[1])
        big_uv[0:ny0, 0:nx0] = big_uv0[0:ny0, 0:nx0]

        dirty_arr = fast_conv(target_img_grey, big_uv)

        cv.imshow("Antenna Layout", frame)
        cv.imshow("Observed Image", dirty_arr)
        cv.imshow(uvwindowname, uv_baselines)

        # Key command handling
        k = stdscr.getch()
        if k != -1:
            stdscr.addstr('Controls: Quit (q), Pause (p), Toggle Persistence (r)\n')
            if k == ord('q'):
                break
            elif k == ord('p'):
                stdscr.addstr('Paused. Press any key to continue...\n')
                while True:
                    k2 = stdscr.getch()
                    if k2 != -1:
                        stdscr.addstr('Unpausing...\n')
                        break
                    cv.waitKey(1)
                    time.sleep(0.1)
            elif k == ord('r'):
                persistence = not persistence
                stdscr.addstr(f"Persistence: {'on' if persistence else 'off'}\n")
            elif k == ord('w'):
                window = not window
                stdscr.addstr(f"Window: {'on' if window else 'off'}\n")
            elif k == ord('g'):
                gaussian = not gaussian
                stdscr.addstr(f"Gaussian: {'on' if gaussian else 'off'}\n")

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    o = optparse.OptionParser()
    o.set_usage('%prog [options]')
    o.set_description(__doc__)
    o.add_option('-i', '--input', dest='input', default=None,
                 help="Input 'sky' image, Default: astro_test_image.jpg")
    o.add_option('-c', '--camera', dest='camera', default=1, type='int',
                 help='Camera device ID in /dev/video*, Default: 1`')
    o.add_option('-r', '--res', dest='res', default=1, type='int',
                 help='Resolution factor, increase this value to decrease the resolution, Default: 1')
    opts, args = o.parse_args(sys.argv[1:])
    curses.wrapper(main, opts, args)
