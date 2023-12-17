from torchvision.io.image import read_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
# from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
import time
img = read_image("office.jpg")

# Step 1: Initialize model with the best available weights
weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9)


model.eval()

# Step 2: Initialize the inference transforms
print(" Step 2: Initialize the inference transforms")
preprocess = weights.transforms()

start_time = time.time()
print("Step 3: Apply inference preprocessing transforms")
# Step 3: Apply inference preprocessing transforms
batch = [preprocess(img)]
print("Step 4: Use the model and visualize the prediction")
prediction = model(batch)[0]

end_time = time.time()
execution_time = end_time - start_time
print("execution_time",execution_time)


print("Step 5: Draw")

labels = [weights.meta["categories"][i] for i in prediction["labels"]]
box = draw_bounding_boxes(img, boxes=prediction["boxes"],
                          labels=labels,
                          colors="red",
                          width=4, font_size=30)
im = to_pil_image(box.detach())
im.show()