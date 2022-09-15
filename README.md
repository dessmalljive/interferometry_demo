### Interferometry demo used at JIVE outreach events ###

Based on https://github.com/adampbeardsley/liveInterferometer

Uses Open CV image detection
Updated for use at JIVE during outreach events by IvB 
31 May 2022

* Operating instructions
Install OpenCV on the MacBook using pip3, python 3.8.3 and MacOS
11.4   
`> pip3 install opencv-python`

The default Facetime camera on MacOS has device number 0, changed the
default input to that, so this should work on Macs now. Not tested
with an external camera.  

Changed the display of the images to *not* show the input image  

Fixed some indents which were not properpy done  

From the terminal launch this with `python3 school_demo.py`

To include another image, first check its dimentions (number of pixels
on each side). Then do:   
`python3 school_demo.py -i <image.jpg> --xsize <x-dimensions> --ysize
<y-dimentions>`  
E.g. `python3 school_demo.py -i test.jpg --xsize 1280 --ysize 720`
