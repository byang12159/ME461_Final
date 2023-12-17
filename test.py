import torch
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image, ImageDraw

def load_model():
    # Load the pre-trained Faster R-CNN model with a ResNet backbone
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    return model

def detect_objects(model, image_path):
    # Load and preprocess the image
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(img).unsqueeze(0)

    # Make the prediction
    with torch.no_grad():
        prediction = model(img_tensor)

    return prediction

def draw_boxes(image, prediction):
    # Draw bounding boxes on the image
    draw = ImageDraw.Draw(image)
    boxes = prediction[0]['boxes']
    for box in boxes:
        draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red", width=3)

    return image

if __name__ == "__main__":
    # Load the pre-trained model
    model = load_model()

    # Provide the path to your sample image
    image_path = "office.jpg"

    # Run object detection
    prediction = detect_objects(model, image_path)

    # Draw bounding boxes on the image
    original_image = Image.open(image_path)
    image_with_boxes = draw_boxes(original_image.copy(), prediction)

    # Display or save the result
    image_with_boxes.show()
    # Optionally, save the result to a file
    image_with_boxes.save("path/to/your/output/result.jpg")

