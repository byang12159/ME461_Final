import time
import torch
import numpy as np
import cv2

img = cv2.imread("office.jpg")

print("original image shape",img.shape)

new_width = 480
new_height = 270
downsample = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
print("downsample image shape",downsample.shape)
# cv2.imshow("office",img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
cv2.imwrite(f"office_downsample{new_width}_{new_height}.jpg",downsample)

