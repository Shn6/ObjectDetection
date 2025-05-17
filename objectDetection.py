"""
Arda ŞAHİN 
"""

import torch  # PyTorch for tensor operations and deep learning
import torchvision # Torchvision for pre-trained models and vision utilities
from torchvision.transforms import functional as F # For converting images to tensors
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights # For model weights
import cv2

# Set device to GPU if available, otherwise CPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load the model with default COCO weights
weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
model.to(device)  # Move the model to the selected device
model.eval()      # Set the model to evaluation mode

# COCO category labels
COCO_INSTANCE_CATEGORY_NAMES = weights.meta["categories"]

# Object detection function
def detect_objects(img, threshold=0.8):

    # Convert the input image (NumPy array) to a PyTorch tensor
    img_tensor = F.to_tensor(img).to(device)

    # Perform inference without computing gradients
    with torch.no_grad():
        outputs = model([img_tensor])[0]  # Get the predictions for the image
    

    # Extract boxes, labels, and scores from the model's output
    boxes, labels, scores = outputs['boxes'], outputs['labels'], outputs['scores']
    results = []

    # Filter predictions based on a threshold
    for box, label, score in zip(boxes, labels, scores):
        if score > threshold:
            results.append({
                'box': box.to('cpu').numpy().astype(int),                     # Convert the box to NumPy format
                'label': COCO_INSTANCE_CATEGORY_NAMES[label.item()],          # Get the category name
                'score': float(score)                                         # Convert the score to a float
            })
    return results  # Return the list of detected objects


image_path = 'detect.jpg'           # Load image from file
frame = cv2.imread(image_path)      # Read the image using OpenCV

# Raise error if image could not be loaded
if frame is None:
    raise FileNotFoundError(f"Could not load image at {image_path}")

# Convert BGR to RGB
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB


detections = detect_objects(rgb, threshold=0.7)  # Run detection


for obj in detections:
    x1, y1, x2, y2 = obj['box']                                              # Get the coordinates of the bounding box
    label = obj['label']                                                     # Get the name of the detected object
    score = obj['score']                                                     # Get the score
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)                 # Draw the green bounding box
    cv2.putText(frame, f'{label}: {score:.2f}', (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)               # Display the label and score above the bounding box



cv2.imshow("Image Object Detection", frame) # Display the final image with detections in a window


cv2.waitKey(0)                              # Wait for any key press to close the window                
cv2.destroyAllWindows()                     # Destroy all OpenCV windows
