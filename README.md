### Interferometry demo used at JIVE outreach events ###

Based on https://github.com/adampbeardsley/liveInterferometer  
Updated for use at JIVE during outreach events by IvB 31 May 2022

Uses Open CV image detection  

*Operating instructions*  
Requires pip3, python 3.8.3 and has been tested on MacOS 11.4   
Install OpenCV on the MacBook `pip3 install opencv-python`  
From the terminal launch this with `python3 school_demo.py`  

To include another image, first check its dimensions (number of pixels
on each side). Then do:   
`python3 school_demo.py -i <image.jpg> --xsize <x-dimensions> --ysize
<y-dimentions>`  
E.g.: `python3 school_demo.py -i test.jpg --xsize 1280 --ysize 720`  
In case of an error, reduce the pixel size of the image and try again,
this can require some fine tuning to get it right, especially with
larger images.

*Changelog*  
* The default Facetime camera on MacOS has device number 0, changed the
default input to that, so this should work on Macs now. Not tested
with an external camera.  
* Changed the display of the images to *not* show the input image  
* Fixed some indents which were not properpy done  


