import time
import torch
import numpy as np
from torchvision import models, transforms
import cv2
from PIL import Image

torch.backends.quantized.engine = 'qnnpack'

img = cv2.imread("office.jpg")
print("image shape",img.shape)

new_width = 270
new_height = 480
img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
print("downsample image shape",img.shape)


preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

net = models.quantization.mobilenet_v2( quantize=True)
# jit model to take it from ~20fps to ~30fps
net = torch.jit.script(net)


last_logged = time.time()
frame_count = 0

sumtime = 0 

with torch.no_grad():
    for i in range(20):
        # read frame
        image = img

        # convert opencv output from BGR to RGB
        image = image[:, :, [2, 1, 0]]
        permuted = image

        # preprocess
        input_tensor = preprocess(image)

        # create a mini-batch as expected by the model
        input_batch = input_tensor.unsqueeze(0)

        started = time.time()
        # run model
        output = net(input_batch)
        now = time.time()
        # do something with output ...
        print("time",now-started)
        sumtime+= now-started

print("average time",sumtime/20)


       