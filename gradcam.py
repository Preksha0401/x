# ==========================================
# DRIVER CODE – IMPORT LIBRARIES
# ==========================================

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2

from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights


# ==========================================
# DRIVER CODE – LOAD PRETRAINED RESNET50
# ==========================================

weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
model.eval()

preprocess = weights.transforms()
categories = weights.meta["categories"]


# ==========================================
# MAIN LOGIC – IMPLEMENT GRAD-CAM
# ==========================================

def generate_gradcam(image_path):

    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0)

    # Variables to store feature maps and gradients
    activations = None
    gradients = None

    # Forward hook: capture feature maps
    def forward_hook(module, input, output):
        nonlocal activations, gradients

        activations = output

        # Capture gradients during backward pass
        output.register_hook(
            lambda grad: store_gradient(grad)
        )

    def store_gradient(grad):
        nonlocal gradients
        gradients = grad

    # Register hook on final convolutional block
    target_layer = model.layer4[-1]
    handle = target_layer.register_forward_hook(forward_hook)

    # Forward pass
    model.zero_grad()
    output = model(input_tensor)

    # Get predicted class
    class_index = output.argmax(dim=1).item()
    class_score = output[0, class_index]

    # Backward pass
    class_score.backward()

    # Remove hook after use
    handle.remove()

    # Calculate importance weights by averaging gradients
    weights = gradients.mean(dim=(2, 3), keepdim=True)

    # Weighted sum of feature maps
    cam = (weights * activations).sum(dim=1, keepdim=True)

    # Apply ReLU
    cam = F.relu(cam)

    # Normalize heatmap
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

    # Resize heatmap to original image size
    cam = F.interpolate(
        cam,
        size=(image.height, image.width),
        mode="bilinear",
        align_corners=False
    )

    heatmap = cam.squeeze().detach().numpy()

    # Convert heatmap to color
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Convert original image to numpy array
    original = np.array(image)

    # Overlay heatmap on original image
    overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

    # Display prediction and Grad-CAM
    print("Predicted class:", categories[class_index])
    print("Confidence:", torch.softmax(output, dim=1)[0, class_index].item())

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(original)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(heatmap)
    plt.title("Grad-CAM Heatmap")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title("Grad-CAM Overlay")
    plt.axis("off")

    plt.show()


# ==========================================
# DRIVER CODE – RUN ON FOUR IMAGES
# ==========================================

image_paths = [
    "piggy_bank.jpg",
    "two_dogs.jpg",
    "bee_eater.jpg",
    "golden_retriever.jpg"
]

for path in image_paths:
    generate_gradcam(path)