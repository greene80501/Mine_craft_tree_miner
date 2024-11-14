import mss
import numpy as np
import cv2
import pyautogui
import torch
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.augmentations import letterbox
import sys
import os
import time

# Add YOLOv5 directory to Python path
sys.path.append(os.path.join(os.getcwd(), "yolov5"))

# Directory to save labeled images
output_dir = "labeled_images"
os.makedirs(output_dir, exist_ok=True)

# Load YOLOv5 model
model_path = 'C:/Users/green/Wood_dectector/yolov5/runs/train/exp2/weights/best.pt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DetectMultiBackend(model_path, device=device)

# Function to preprocess the screen for YOLOv5
def preprocess_image(img):
    img = letterbox(img, stride=32, auto=True)[0]  # Resize with padding
    img = img[:, :, ::-1].transpose(2, 0, 1)  # Convert to CHW format
    img = np.ascontiguousarray(img)
    return torch.from_numpy(img).to(device).float() / 255.0

# Function to capture the screen
def capture_screen():
    with mss.mss() as sct:
        monitor = sct.monitors[1]  # Capture primary monitor
        screen = np.array(sct.grab(monitor))
        return cv2.cvtColor(screen, cv2.COLOR_BGRA2BGR)

# Function to detect the closest wood (largest bounding box)
def detect_closest_wood(screen):
    input_image = preprocess_image(screen)
    input_image = input_image.unsqueeze(0)  # Add batch dimension
    predictions = model(input_image)
    predictions = non_max_suppression(predictions, conf_thres=0.5, iou_thres=0.45)

    if predictions[0] is not None and len(predictions[0]):
        largest_box = None
        largest_area = 0
        for det in predictions[0]:
            x1, y1, x2, y2 = det[:4]
            area = (x2 - x1) * (y2 - y1)
            if area > largest_area:
                largest_area = area
                largest_box = [int(x1), int(y1), int(x2), int(y2)]
        return largest_box
    return None

# Function to save labeled images
def save_labeled_image(screen, box, action, frame_count):
    labeled_screen = screen.copy()

    # Draw bounding box if detected
    if box is not None:
        x1, y1, x2, y2 = box
        cv2.rectangle(labeled_screen, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
        cv2.putText(
            labeled_screen,
            "Wood",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    # Add action text
    cv2.putText(
        labeled_screen,
        action,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
    )

    # Save image
    output_path = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
    cv2.imwrite(output_path, labeled_screen)

# Function to move toward detected wood
def move_to_wood(box, screen_width, screen_height):
    x1, y1, x2, y2 = box
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
    screen_center_x, screen_center_y = screen_width // 2, screen_height // 2
    action = "Already aligned with wood"

    # Move horizontally
    if center_x > screen_center_x + 20:
        action = "Move right"
        pyautogui.keyDown('d')
        time.sleep(0.1)
        pyautogui.keyUp('d')
    elif center_x < screen_center_x - 20:
        action = "Move left"
        pyautogui.keyDown('a')
        time.sleep(0.1)
        pyautogui.keyUp('a')
    
    # Move vertically
    if center_y > screen_center_y + 20:
        action = "Move backward"
        pyautogui.keyDown('s')
        time.sleep(0.1)
        pyautogui.keyUp('s')
    elif center_y < screen_center_y - 20:
        action = "Move forward"
        pyautogui.keyDown('w')
        time.sleep(0.1)
        pyautogui.keyUp('w')

    # Jump if the wood is significantly above the player
    if center_y < screen_center_y - 50:
        action = "Jumping"
        pyautogui.keyDown('space')
        time.sleep(0.1)
        pyautogui.keyUp('space')

    return action

# Function to mine the wood
def mine_wood():
    print("Mining wood...")
    pyautogui.mouseDown()
    time.sleep(2)
    pyautogui.mouseUp()

# Main function
def main():
    frame_count = 0
    with mss.mss() as sct:
        screen_width = sct.monitors[1]['width']
        screen_height = sct.monitors[1]['height']
        
        while True:
            # Capture the screen
            screen = capture_screen()
            
            # Detect the closest wood
            box = detect_closest_wood(screen)
            
            if box is not None:
                action = move_to_wood(box, screen_width, screen_height)
                mine_wood()
            else:
                action = "No wood detected. Searching..."
            
            # Save labeled image
            save_labeled_image(screen, box, action, frame_count)
            frame_count += 1

            print(action)
            time.sleep(0.1)  # Add a small delay

if __name__ == "__main__":
    main()
