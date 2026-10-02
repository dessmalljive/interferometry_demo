#! /usr/bin/python3
"""
Interferometer/PSF simulator

Created by: Jack Hickish
Minor Modifications by: Griffin Foster
Modified 14/06/16 -- Added command entry with curses. Added pause and "persistence" mode
Modified Jan 16th 2019 -- CJ, Edgar, + Adam edit. Plot 3 images in one window.
                          Could not get last one to show in same window
                          Attempts saved in "camera_window_attempt"

TODO: add color
TODO: adjust detection paramters
TODO: add rotation command
"""

### IVB adjustments for school demo in JIVE
### 20220411
## Installed OpenCV on the MacBook using pip3, python 3.8.3 and MacOS 11.4
### $> pip3 install opencv-python
## The default Facetime camera on MacOS has device number 0, changed the default input to that
## Changed the display of the images to not show the input image
## Fixed some indents which were not properly done
## From the terminal launch this with > python3 school_demo.py
## To include another image, first check the pixel dimentions
## Then do > python3 school_demo.py -i <image.jpg> --xsize <x-dimensions> --ysize <y-dimentions>
## E.g. python3 school_demo.py -i test.jpg --xsize 1280 --ysize 720


# For ubuntu 12.04 install see: http://karytech.blogspot.com/2012/05/opencv-24-on-ubuntu-1204.html
import numpy as np
import time
import sys
import optparse
import curses
import os
import cv2 as cv
import scipy



def fast_conv(image, uv_tracks):
    # FIXME: We don't currently handle resizing.
    uv_image = np.fft.fft2(image)
    #uv_tracks2 = np.ones(uv_tracks.shape)
    sample = uv_image * uv_tracks
    dirty = np.abs(np.fft.ifft2(sample))
    s = np.max(dirty)
    if s > 0:
        dirty = 255*1/s * dirty
    return dirty


class Mutable(object):
    def __init__(self, v):
        self.v = v
    def set(self, v):
        self.v = v
    def get(self):
        return self.v

def main(stdscr, opts, args):
    # check /dev/, ID is attached to video device (0 is in the internal)
    CAMERA_DEVICE_INDEX = opts.camera
    cam = cv.VideoCapture(CAMERA_DEVICE_INDEX)
    if opts.input is None:
        infile = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                              'astro_test_image.jpg')
        target_image = cv.imread(infile)
    else:
        target_image = cv.imread(opts.input)

    # The sizes of the image we are trying to reconstruct
    ysize, xsize = target_image.shape[:2]
    aspect = xsize/ysize
    target_img_grey = cv.cvtColor(target_image, cv.COLOR_BGR2GRAY)
    #
    brightness_threshold = 100
    target_img_grey[target_img_grey < brightness_threshold] = 0

    RESCALE_FACTOR =  opts.res

    persistence = False
    window = False
    gaussian = True
    # FIXME: My guess is the psf should be much smaller now that baseline support isn't just wrong
    # psf = PSF(xsize, ysize, persistence, window)

    stdscr.nodelay(1)  # don't wait for keyboard input when calling getch
    stdscr.clear()
    stdscr.addstr('Controls: Quit (q), Pause (p), Toggle Persistence (r)\n')

    # Sigh
    rv, layout_img = cam.read()
    yc, xc = layout_img.shape[:2]
    ac = xc/yc
    if ac > aspect:
        xc = int(np.around(aspect * yc))
    elif ac < aspect:
        yc = int(np.around(xc / aspect))
    layout_img = layout_img[:yc, :xc, :]
    hour_angle = Mutable(1)
    hour_angle.set(3)

    uvwindowname = "UV tracks"
    cv.namedWindow(uvwindowname)
    trackbar_name = "Hour angle 12"
    cv.createTrackbar(trackbar_name, uvwindowname, 0, 12, hour_angle.set)


    while(True):
        # Grab 4 images. The images are buffered, so this helps
        # get one that is relatively recent. My cv doesn't
        # seem to support reducing the size of the buffer.
        imgs = []
        for i in range(4):
            rv, layout_img = cam.read()

        layout_img = cv.resize(layout_img, (xsize, ysize), interpolation=cv.INTER_CUBIC)
        layout_img_grey = cv.cvtColor(layout_img, cv.COLOR_BGR2GRAY)
        
        # Match the aspect ration of the webcam to the image we are using
        yc, xc = layout_img.shape[:2]
        ac = xc/yc
        if ac > aspect:
            xc = int(np.around(aspect * yc))
        elif ac < aspect:
            yc = int(np.around(xc / aspect))
        layout_img = layout_img[:yc, :xc, :]

        # stdscr.addstr("{} {} {}\n".format(aspect, ac, layout_img.shape))

        #blursize = 9
        blursize = 5        
        layout_img_grey = cv.medianBlur(layout_img_grey, blursize, blursize)

        # set the station locations to zero for each loop, unless we have persistence, in which
        # case leave the ones from the previous iteration in
        if not persistence:
            station_overlay = np.zeros_like(layout_img_grey)

        method = cv.HOUGH_GRADIENT  # only method that exists

        # If it is too large, some circles may be missed.
        circles = None  # set to None
        # param1 = 50  # threshold parameterq
        # param2 = 30  # The smaller it is, the more false circles may be detected.
        # param1, param2 = 100.0, 100.0 # defaults
        param1, param2 = 50, 30
        minRadius = 10  # Minimum circle radius
        maxRadius = 20  # Maximum circle radius
        # Minimum distance between the centers of the detected circles.
        # If the parameter is too small, multiple neighbor circles may be falsely detected
        # in addition to a true one.
        minDist = 1.5 * minRadius
        dp = 2 # Inverse ratio of the accumulator resolution to the image resolution

        circles = cv.HoughCircles(layout_img_grey, method, dp, minDist, circles,
                                   param1, param2, minRadius, maxRadius)

        station_locations = []
        # For diagnostics, we draw circles with their radius
        ds = 5 # Size of rectange we use for antenna in U-V space
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for x, y, r in circles[0]:
                x = int(x)
                y = int(y)
                if x > xsize or y > ysize:
                    pass
                station_overlay[x-ds:x+ds,  y-ds:y+ds] = 1
                # 255 is *white* here
                cv.circle(layout_img_grey, (x, y), r, 255, 1)
                cv.circle(layout_img_grey, (x, y), 2, 255, 2)
                station_locations.append((y, x))

        # Create UV tracks for a given hour angle
        hour_angle_rad = 2*np.pi*hour_angle.get()/24 # 0.26 per hour
        fails2 = 0
        uv_baselines = np.zeros([ysize // RESCALE_FACTOR, xsize // RESCALE_FACTOR])
        for i in range(len(station_locations)):
            for j in range(0, i):
                y1, x1 = station_locations[i]
                y2, x2 = station_locations[j]
                dx = (x2 - x1) // RESCALE_FACTOR
                dy = (y2 - y1) // RESCALE_FACTOR
                ha0 = np.arctan2(dx, dy)
                r = np.sqrt(dy**2 + dx**2)
                Y, X = uv_baselines.shape
                angles = np.arange(0, hour_angle_rad, 0.005) # Magic number!
                try:
                    uv_baselines[(Y + dy)//2, (X + dx)//2] = 1
                    uv_baselines[(Y - dy)//2, (X - dx)//2] = 1
                    for ha in angles:
                        alpha = ha0 + ha
                        ay = r*np.cos(alpha)
                        ax = r*np.sin(alpha)
                        cx = round(X + ax)//2
                        cy = round(Y + ay)//2
                        uv_baselines[cx-2:cx+2, cy-2:cy+2] = 1
                        cx = round(X - ax)//2
                        cy = round(Y - ay)//2
                        uv_baselines[cx-2:cx+2, cy-2:cy+2] = 1
                except Exception as e:
                    print("Exception: {}".format(e))
                    fails2 += 1
        Y, X = uv_baselines.shape
        # uv_baselines[X//2-10:X//2+10, Y//2-10:Y//2+10] = 1
        if False:
            margin = 2
            uv_baselines[0:margin, :] = 1
            uv_baselines[Y-margin:, :] = 1
            uv_baselines[:, 0:margin] = 1
            uv_baselines[:, X-margin:] = 1
        
        
        
        # I guess this is supposed to be a normalized psf? (Whatever that is)
        # psf_norm = psf.getNorm(uv_baselines, window, persistence, gaussian)
        # Note: uv_stations isn't used after this.


        # This is impressively irritating!
        big_uv = np.zeros(target_img_grey.shape, float)
        big_uv0 = scipy.ndimage.zoom(uv_baselines, opts.res)
        nx0, ny0 = big_uv0.shape
        big_uv[0:nx0, 0:ny0] = big_uv0
        dirty_arr = fast_conv(target_img_grey, big_uv)

        if dirty_arr.max() != 0:
            dirty_arr /= dirty_arr.max()

        # stacking images into one window instead of 4. one camera will not work; see note at top
        # numpy_horiz_concat1 = np.concatenate((target_img_grey, dirty_arr), axis=1)
        # numpy_horiz_concat2 = np.concatenate((layout_img_grey, psf_norm), axis=1)		
	
        #cat1rs = cv.resize(numpy_horiz_concat1, (1200, 500))
        #cat2rs = cv.resize(numpy_horiz_concat2, (1200, 500))

        # numpy_vert_concat= np.concatenate((cat1rs, cat2rs), axis=0)
        # showing windows
        # cv.imshow("Target, Observed, PSF", numpy_vert_concat)
        cv.imshow("Antenna Layout", layout_img_grey)
        # cv.imshow("Target Image", target_img_grey)
        # cv.imshow("horiz concat", numpy_horiz_concat2)
        # cv.imshow("Point Spread", psf_norm)
        # cv.imshow("Antennas", uv_stations*255)
        # cv.imshow(uvwindow, uv_baselines)
        cv.imshow("Observed Image", dirty_arr)
        cv.imshow(uvwindowname, uv_baselines)

        # Key command handling
        k = stdscr.getch()
        if k != -1:
            # stdscr.clear()
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
                    time.sleep(0.25)
            elif k == ord('r'):
                if persistence:
                    stdscr.addstr('Turning Persistence off\n')
                else:
                    stdscr.addstr('Turning Persistence on\n')
                persistence = not persistence
            elif k == ord('w'):
                if window:
                    stdscr.addstr('Turning window off\n')
                else:
                    stdscr.addstr('Turning window on\n')
                window = not window
            elif k == ord('g'):
                if gaussian:
                    stdscr.addstr('Turning gaussian off\n')
                else:
                    stdscr.addstr('Turning guassian on\n')
                gaussian = not gaussian
        cv.waitKey(1)

cv.destroyAllWindows()

if __name__ == "__main__":
    o = optparse.OptionParser()
    o.set_usage('%prog [options]')
    o.set_description(__doc__)
    o.add_option('-i', '--input', dest='input', default=None,
                 help='Input \'sky\' image, Default: astro_test_image.jpg')
    o.add_option('-c', '--camera', dest='camera', default=0, type='int',
                 help='Camera device ID in /dev/video*, Default: 1')
    o.add_option('-r', '--res', dest='res', default=4, type='int',
                 help='Resolution factor, increase this value to decrease the'
                      'resolution, Default: 4')
    opts, args = o.parse_args(sys.argv[1:])
    curses.wrapper(main, opts, args)
