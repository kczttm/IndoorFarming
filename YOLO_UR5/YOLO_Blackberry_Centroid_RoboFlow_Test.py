import io
import cv2
import requests
from PIL import Image
import numpy as np
from requests_toolbelt.multipart.encoder import MultipartEncoder


def get_blackberry_centroid(frame):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pilImage = Image.fromarray(image)
    buffered = io.BytesIO()
    pilImage.save(buffered, quality=100, format="JPEG")
    m = MultipartEncoder(fields={'file': ("imageToUpload", buffered.getvalue(), "image/jpeg")})
    response = requests.post("https://detect.roboflow.com/artificial_blackberry/1?api_key=P3D9VxtH5FBcHJD4KoMd", data=m, headers={'Content-Type': m.content_type})
    r = response.json()
    print(r)
    center_x = r['predictions'][0]['x']
    center_y = r['predictions'][0]['y']
    return np.array([int(center_x),int(center_y)])

image = cv2.imread("test_1.jpg")
cv2.imshow("",image)
get_blackberry_centroid(image)