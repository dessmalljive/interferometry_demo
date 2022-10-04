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
## Fixed some indents which were not properpy done
## From the terminal launch this with > python3 school_demo.py
## To include another image, first check the pixel dimentions
## Then do > python3 school_demo.py -i <image.jpg> --xsize <x-dimensions> --ysize <y-dimentions>
## E.g. python3 school_demo.py -i test.jpg --xsize 1280 --ysize 720


# For ubuntu 12.04 install see: http://karytech.blogspot.com/2012/05/opencv-24-on-ubuntu-1204.html
import cv2 as cv
import numpy as np
import time
import sys
import optparse
import curses
import os



# Should probably just be a method of the PSF class now.
def fast_conv(image, psf):
    max_size = np.array([np.max([image.shape[0], psf.shape[0]]),
                         np.max([image.shape[1], psf.shape[1]])])
    n = int(2**np.ceil(np.log2(max_size[0])))
    m = int(2**np.ceil(np.log2(max_size[1])))
    imageDirty = np.fft.irfft2(np.fft.rfft2(image, (n, m)) * np.fft.rfft2(psf, (n, m)))
    return imageDirty[psf.shape[0] // 2:image.shape[0] + psf.shape[0] // 2,
                      psf.shape[1] // 2:image.shape[1] + psf.shape[1] // 2]

class PSF(object):
    def __init__(self, xsize, ysize, persistence=False, window=False):
        self.xsize = xsize
        self.ysize = ysize
        
        # make a 2D Gaussian to modulate the PSF with
        def gauss2d(x0, y0, amp, stdx, stdy):
            return lambda x, y: amp * np.exp(-1. * ((((x - x0)**2.) / (2 * stdx**2.)) +
                                                    (((y - y0)**2.) / (2 * stdy**2.))))
        gaussFunc = gauss2d(0., 0., 1., 40., 40.)
        xx = np.arange(self.xsize) - (self.xsize / 2)
        yy = np.arange(self.ysize) - (self.ysize / 2)
        xv, yv = np.meshgrid(xx, yy)
        self.gaussGrid = gaussFunc(xv, yv)
        self.psf = np.zeros([self.ysize, self.xsize])
        
    def getNorm(self, station_locs, persistence=False, window=False, gaussian=True):
        if not persistence:
            self.psf = np.zeros([self.ysize, self.xsize])
            
        self.psf += np.fft.fftshift(np.abs(np.fft.fft2(station_locs, s=[self.ysize, self.xsize]))**2)

        if self.psf.max() != 0:
            if window:
                bls = np.fft.ifft2(np.fft.fftshift(self.psf))
                abs_bls = np.abs(bls)
                abs_bls[0, 0] = 0
                abs_bls[abs_bls < 0.5] = 1e9
                win_psf = np.abs(np.fft.fftshift(np.fft.fft2(bls / abs_bls, s=[self.ysize, self.xsize])))
                psf_norm = (win_psf / win_psf.max())
                # apply Gaussian taper to the PSF
            else:
                psf_norm = (self.psf / self.psf.max())
            if gaussian:
                psf_norm *= self.gaussGrid 
        else:
            psf_norm = self.psf
        return psf_norm



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
        
    RESCALE_FACTOR = opts.res

    persistence = False
    window = False
    gaussian = True
    psf = PSF(xsize, ysize, persistence, window)
    
    stdscr.nodelay(1)  # don't wait for keyboard input when calling getch
    stdscr.clear()
    stdscr.addstr('Controls: Quit (q), Pause (p), Toggle Persistence (r)\n')


    # Sigh
    rv, layout_img = cam.read()
    old_shape = layout_img.shape
    yc, xc = layout_img.shape[:2]
    ac = xc/yc
    if ac > aspect:
        xc = int(np.around(aspect * yc))
    elif ac < aspect:
        yc = int(np.around(xc / aspect))
    layout_img = layout_img[:yc, :xc, :]
    stdscr.addstr("New xc {} yc {}\n".format(xc, yc))
    stdscr.addstr("Old shape: {} Aspect ratio: {} Target aspect: {} New shape: {}\n".format(old_shape, ac, aspect, layout_img.shape))

    


    while(True):
        # Grab 4 images. The images are buffered, so this helps
        # get one that is relatively recent. My cv doesn't
        # seem to support reducing the size of the buffer.
        for i in range(4):
            rv, layout_img = cam.read()

        # Match the aspect ration of the webcam to the image we are using
        yc, xc = layout_img.shape[:2]
        ac = xc/yc
        if ac > aspect:
            xc = int(np.around(aspect * yc))
        elif ac < aspect:
            yc = int(np.around(xc / aspect))
        layout_img = layout_img[:yc, :xc, :]
        
        # stdscr.addstr("{} {} {}\n".format(aspect, ac, layout_img.shape))

        layout_img = cv.resize(layout_img, (xsize, ysize), interpolation=cv.INTER_CUBIC)
        layout_img_grey = cv.cvtColor(layout_img, cv.COLOR_BGR2GRAY)
        layout_img_grey = cv.medianBlur(layout_img_grey, 9, 9)

        # stdscr.addstr("{}\n".format(layout_img_grey.shape))
        
        # set the station locations to zero for each loop, unless we have persistence, in which
        # case leave the ones from the previous iteration in
        if not persistence:
            station_overlay = np.zeros_like(layout_img_grey)
            
        station_locs = np.zeros([ysize // RESCALE_FACTOR, xsize // RESCALE_FACTOR])

        method = cv.HOUGH_GRADIENT  # only method that exists
        
        # If it is too large, some circles may be missed.
        circles = None  # set to None
        #param1 = 50  # threshold parameterq
        #param2 = 30  # The smaller it is, the more false circles may be detected.
        # param1, param2 = 100.0, 100.0 # defaults
        param1, param2 = 50, 50
        minRadius = 10  # Minimum circle radius
        # Minimum distance between the centers of the detected circles.
        # If the parameter is too small, multiple neighbor circles may be falsely detected in addition to a true one.
        minDist = 1.5 * minRadius  
        maxRadius = 50  # Maximum circle radius
        dp = 2 # Inverse ratio of the accumulator resolution to the image resolution
        
        circles = cv.HoughCircles(layout_img_grey, method, dp, minDist, circles,
                                   param1, param2, minRadius, maxRadius)

        # For diagnostics, we draw circles with their radius
        ds = 5 # Size of rectange we use for antenna in U-V space
        fails = 0
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for x, y, r in circles[0]:
                station_overlay[x-ds:x+ds,  y-ds:y+ds] = 1
                # 255 is *white* here
                cv.circle(layout_img_grey, (x, y), r, 255, 1)
                cv.circle(layout_img_grey, (x, y), 2, 255, 2)
                try:
                    station_locs[y // RESCALE_FACTOR, x // RESCALE_FACTOR] = 1
                except:
                    fails += 1
                    
        ## draw white squares at the station locations
        # layout_img_grey[station_overlay == 1] = 128
                
        psf_norm = psf.getNorm(station_locs, window, persistence, gaussian)
        dirty_arr = fast_conv(target_img_grey, psf_norm)

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
        cv.imshow("Point Spread", psf_norm)
        cv.imshow("Antennas", station_locs*255)
        cv.imshow("Observed Image", dirty_arr)

        # Key command handling
        k = stdscr.getch()
        if k != -1:
            # stdscr.clear()
            stdscr.addstr('Controls: Quit (q), Pause (p), Toggle Persistence (r)\n')
            if k == ord('q'):
                break
            elif k == ord('p'):
                stdscr.addstr('Paused. Press any key to continue...\n')
                while(True):
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
