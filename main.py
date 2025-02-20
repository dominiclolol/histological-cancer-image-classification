import torch
import torch.nn.functional as F
import os
import cv2
from PIL import Image
from model import transform_TumorClassifierb4, TumorClassifierb4

model = TumorClassifierb4(num_classes=2)
model.load_state_dict(torch.load('40X.pth'))
model.eval()

input_folder = 'BreaKHis_v1-data-for-use/testing/40X/benign' #Path to folder containing images for analysis
image_files = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
class_labels = ["Benign", "Malignant"]
output_file = "predictions.txt"

with open(output_file, "w") as f:
    f.write("Predictions with Model Confidence Scores:\n")
       
for filename in image_files:
    input_path = os.path.join(input_folder, filename)
    img = cv2.imread(input_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img_tensor = transform_TumorClassifierb4(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)
        confidence, pred = probs.max(dim=1)

    with open(output_file, "a") as f:
        f.write(f"{filename} - {class_labels[pred.item()]} - {confidence.item() * 100:.2f}%\n")
