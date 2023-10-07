import torch
import clip
from PIL import Image
from sklearn.cluster import KMeans
import numpy as np
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device, download_root="/misc/student/sharmaa/modelcache")
image_in = Image.open("/misc/student/sharmaa/groupvit/GroupViT/demo/examples/coco.jpg") #.convert("RGB")
image = preprocess(image_in).unsqueeze(0).to(device)
text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    print(image_features.shape, text_features.shape)
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    # Perform K-means clustering
image_features = image_features.cpu().numpy()
k = 10  # Number of clusters
kmeans = KMeans(n_clusters=k)
kmeans.fit(image_features.reshape(-1, 1))

# Get the cluster labels for each image feature
cluster_labels = kmeans.labels_
print(cluster_labels)

# Project features back to the image
image_features_projected = kmeans.cluster_centers_[cluster_labels].reshape(image_in.size[1], image_in.size[0])

# Convert the projected features to image format
image_features_projected = (image_features_projected * 255).astype(np.uint8)
image_features_projected = Image.fromarray(image_features_projected)

# Display the projected features image
image_features_projected.show()
print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]