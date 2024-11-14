import torch
from PIL import Image
from pathlib import Path

# Path to your YOLOv5 model
model_path = 'C:/Users/green/Wood_dectector/yolov5/runs/train/exp2/weights/best.pt'  # Update this with your model's path
# Path to your image
image_path = 'C:/Users/green/Wood_dectector/test/images/2022-06-22_13-38-11_png.rf.076451e2963f91e9ae3233f07cf60626.jpg'  # Update this with your image's path
# Output directory
output_dir = 'C:/Users/green/Wood_dectector/img_test'

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)

# Perform inference on the image
results = model(image_path)

# Save labeled image
results.save(save_dir=output_dir)

# Display results
print(f"Labels and coordinates: {results.xyxy[0]}")  # xyxy format: [x_min, y_min, x_max, y_max, confidence, class]

# Optionally, display the image with bounding boxes (requires OpenCV or PIL)
img = Image.open(Path(output_dir) / 'exp' / Path(image_path).name)
img.show()
