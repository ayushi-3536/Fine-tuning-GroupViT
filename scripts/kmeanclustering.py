import torch
import torchvision.transforms as transforms
from PIL import Image
from sklearn.cluster import KMeans
import numpy as np
from transformers import CLIPModel, CLIPProcessor
import os
# Set the cache directory
cache_dir = "/misc/student/sharmaa/modelcache"
os.environ["HF_HOME"] = cache_dir
# Load the CLIP model
model_name = 'openai/clip-ViT-B-32'
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

# Set the device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load and preprocess the image
image_path = "path_to_your_image.jpg"  # Replace with the path to your image
image = Image.open(image_path).convert("RGB")
inputs = processor(images=image, return_tensors="pt", padding=True)
image_input = {k: v.to(device) for k, v in inputs.items()}

# Get the image features from CLIP
with torch.no_grad():
    image_features = model.get_image_features(**image_input).cpu().numpy()

# Perform K-means clustering
k = 5  # Number of clusters
kmeans = KMeans(n_clusters=k)
kmeans.fit(image_features.reshape(-1, 1))

# Get the cluster labels for each image feature
cluster_labels = kmeans.labels_

# Project features back to the image
image_features_projected = kmeans.cluster_centers_[cluster_labels].reshape(image.size[1], image.size[0])

# Convert the projected features to image format
image_features_projected = (image_features_projected * 255).astype(np.uint8)
image_features_projected = Image.fromarray(image_features_projected)

# Display the projected features image
image_features_projected.show()
