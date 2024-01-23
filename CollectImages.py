from picamera2 import Picamera2, Preview
import time
import os
import random
from pathlib import Path
import cv2 as cv

camera = Picamera2()
camera_configuration = camera.create_still_configuration(main={"size": (640, 480)},
                                                         lores = {"size": (640, 480)},
                                                         display = "lores")
camera.configure(camera_configuration)

camera.start_preview(Preview.QTGL)
camera.start()
time.sleep(2)

while True:
    captured_image = camera.capture_array()
    image_index = random.randint(100000, 500000)
    adress = "/home/maze/ProjektR/" + str(image_index) + ".jpg"
    while Path(adress).is_file():
        image_index = random.randint(100000, 500000)
        adress = "/home/maze/ProjektR/" + str(image_index) + ".jpg"
        
    cv.imwrite(adress, captured_image)
    time.sleep(1)
    
