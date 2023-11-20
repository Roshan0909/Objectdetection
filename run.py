import torch
import cv2
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import yolov5

# Load YOLOv5 object detection model
model_yolo = yolov5.load('yolov5s.pt')

# Load BART text generation model
tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')
model_bart = AutoModelForSeq2SeqLM.from_pretrained('facebook/bart-base')

# List of image paths
image_paths = [
    r"C:\Users\ROSHAN\Desktop\55267872-old-basketball-colorful-is-red-yellow-green-on-the-grass.jpg",
    # Add more image paths if needed
]

# Detect objects and generate summaries for each image
for image_path in image_paths:
    # Load image
    image = cv2.imread(image_path)

    # Detect objects using YOLOv5
    results = model_yolo(image)

    # Check if there are detections
    if isinstance(results.pred, torch.Tensor) and results.pred.size(0) > 0:  # Check if results.pred is a non-empty tensor
        # Extract bounding boxes, confidences, and class indices
        bounding_boxes = results.pred[:, :4]  # Extracting bounding box coordinates [x1, y1, x2, y2]
        confidences = results.pred[:, 4]  # Confidence scores
        class_indices = results.pred[:, 5]  # Class indices

        # Assuming you have a list of class labels
        class_labels = ["class1", "class2", "class3", ...]

        # Generate object descriptions
        object_descriptions = []
        for i in range(len(bounding_boxes)):
            bounding_box = bounding_boxes[i]
            confidence = confidences[i].item()  # Convert confidence to Python scalar
            class_index = int(class_indices[i].item())  # Convert class index to Python scalar

            # Access class label using the index
            class_label = class_labels[class_index] if class_index < len(class_labels) else 'Unknown'

            # Create object description
            object_description = f"A {class_label} is located at coordinates ({bounding_box[0]}, {bounding_box[1]}) to ({bounding_box[2]}, {bounding_box[3]}) with confidence {confidence}"
            object_descriptions.append(object_description)

        # Generate summary using BART
        summary_inputs = tokenizer.batch_encode_plus(object_descriptions, return_tensors="pt", padding=True, truncation=True)
        summary_ids = model_bart.generate(**summary_inputs, max_length=64, num_beams=4, length_penalty=2.0, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        # Print results
        print(f"Image: {image_path}")
        print(f"Object Descriptions: {object_descriptions}")
        print(f"Summary: {summary}")
    else:
        print(f"No objects detected in {image_path}")
