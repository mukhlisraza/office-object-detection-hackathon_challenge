import torch
from ultralytics import YOLO

# Load  best-trained model
model = YOLO('path/to/your/best_model.pt')  # Update this path to your model

# path to testing images
test_dir = r'path/to/your/test/images'  # Update this path to your test images directory

# Results will be saved in the same directory
results = model.predict(source=test_dir, plots=True, save=True, conf=0.5, device='cuda' if torch.cuda.is_available() else 'cpu')

print("Inference completed. Check the results directory for runs/detect.")