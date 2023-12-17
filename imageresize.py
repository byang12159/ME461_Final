import cv2

def downsample_image(input_path, output_path, scale_factor):
    # Read the image
    original_image = cv2.imread(input_path)

    # Get the original height and width
    original_height, original_width = original_image.shape[:2]

    # Calculate the new height and width after downsampling
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)

    # Resize the image
    downsampled_image = cv2.resize(original_image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Save the downsampled image
    cv2.imwrite(output_path, downsampled_image)

if __name__ == "__main__":
    # Set the paths and scale factor
    input_image_path = "path/to/your/input/image.jpg"
    output_image_path = "path/to/your/output/downsampled_image.jpg"
    scale_factor = 0.5  # Adjust the scale factor as needed

    # Downsample the image
    downsample_image(input_image_path, output_image_path, scale_factor)

    print(f"Image downsampled and saved to {output_image_path}")
