import os
import sys
import cv2
import json

if (sys.platform == "darwin"):
    cwd = os.getcwd()
    cwd = cwd.split("/")
    path_to_downloads = cwd[0:3]
    path_to_downloads.append("Downloads")
    path_to_downloads = "/".join(path_to_downloads)

os.chdir(path_to_downloads)
directory = "blackberry_endoscope_images"
parent_dir = path_to_downloads
path = os.path.join(parent_dir, directory)

if not os.path.exists(path):
    os.mkdir(path)

os.chdir(path)

if os.path.isfile(os.path.join(path, "tracker.json")) is False:
    with open("tracker.json", "w") as outfile:
        json.dump({"next_image_id" : 0}, outfile)

with open('tracker.json', 'r') as openfile:
    tracker = json.load(openfile)

cam = cv2.VideoCapture(0)
cam.set(3, 1280)
cam.set(4, 720)

curr_image = None
for _ in range(10):
    result, curr_image = cam.read()

img_filename = "blackberry_img_"+str(tracker["next_image_id"])+".jpg"
cv2.imwrite(img_filename, curr_image)


tracker["next_image_id"] = tracker["next_image_id"] + 1
with open("tracker.json", "w") as outfile:
    json.dump(tracker, outfile)