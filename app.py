import torch
import cv2
import yolov5
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load YOLOv5 object detection model
model = yolov5.load('yolov5s.pt')

# Load BART text generation model
tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')
model = AutoModelForSeq2SeqLM.from_pretrained('facebook/bart-base')

# Load images
images_paths = [r"C:\Users\ROSHAN\project streamlit\marguerite-729510_640.jpg"]


# Detect objects in each image
for image_path in images_paths:
    # Load image
    image = cv2.imread(image_path)

    # Detect objects
    results = model(image)

    # Generate object descriptions
    object_descriptions = []
    for result in results:
        bounding_box = result['bbox']
        class_label = result['class']
        object_description = f"A {class_label} is located at coordinates ({bounding_box[0]}, {bounding_box[1]}) to ({bounding_box[2]}, {bounding_box[3]})"
        object_descriptions.append(object_description)

    # Generate summary
    summary = model(input_text=object_descriptions, max_length=64)
    print(f"Image: {image_path}")
    print(f"Summary: {summary}")
