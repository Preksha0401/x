# ==========================================
# DRIVER CODE – IMPORT LIBRARIES
# ==========================================

import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2

from PIL import Image
from lime import lime_image
from skimage.segmentation import slic
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
# MAIN LOGIC – PREDICTION FUNCTION FOR LIME
# ==========================================

def predict_images(images):

    batch = []

    for img in images:

        # Convert numpy image to PIL
        pil_img = Image.fromarray(img.astype("uint8"))

        # Preprocess image
        tensor = preprocess(pil_img)

        batch.append(tensor)

    batch = torch.stack(batch)

    # Predict using ResNet50
    with torch.no_grad():
        outputs = model(batch)

        probabilities = torch.softmax(outputs, dim=1)

    return probabilities.numpy()


# ==========================================
# DRIVER CODE – LOAD THREE IMAGES
# ==========================================

image_paths = [
    "indoor_dog.jpg",
    "beach_dog.jpg",
    "forest_dog.jpg"
]

images = []

for path in image_paths:

    img = Image.open(path).convert("RGB")
    img = img.resize((224, 224))

    images.append(np.array(img))


# ==========================================
# DRIVER CODE – DISPLAY PREDICTIONS
# ==========================================

for i, img in enumerate(images):

    prediction = predict_images([img])[0]

    class_index = np.argmax(prediction)

    print("Image", i + 1)
    print("Prediction:", categories[class_index])
    print("Probability:", prediction[class_index])


# ==========================================
# MAIN LOGIC – CREATE LIME EXPLAINER
# ==========================================

explainer = lime_image.LimeImageExplainer()


# ==========================================
# MAIN LOGIC – GENERATE LIME EXPLANATION
# ==========================================

def generate_lime(image, number_of_segments):

    # Segment image using SLIC
    def segmentation_function(img):

        return slic(
            img,
            n_segments=number_of_segments,
            compactness=10,
            sigma=1,
            start_label=0,
            channel_axis=-1
        )

    # Generate LIME explanation
    explanation = explainer.explain_instance(
        image,
        predict_images,
        top_labels=5,
        hide_color=0,
        num_samples=1000,
        segmentation_fn=segmentation_function
    )

    return explanation


# ==========================================
# DRIVER CODE – SELECT ONE IMAGE
# ==========================================

selected_image = images[0]

original_prediction = predict_images([selected_image])[0]

target_class = np.argmax(original_prediction)

print("Original predicted class:", categories[target_class])
print("Original probability:", original_prediction[target_class])


# ==========================================
# DRIVER CODE – GENERATE TWO EXPLANATIONS
# ==========================================

# Explanation A: 50 superpixels
explanation_A = generate_lime(selected_image, 50)

# Explanation B: 150 superpixels
explanation_B = generate_lime(selected_image, 150)


# ==========================================
# MAIN LOGIC – GET IMPORTANT REGIONS
# ==========================================

image_A, mask_A = explanation_A.get_image_and_mask(
    target_class,
    positive_only=True,
    num_features=10,
    hide_rest=False
)

image_B, mask_B = explanation_B.get_image_and_mask(
    target_class,
    positive_only=True,
    num_features=10,
    hide_rest=False
)


# ==========================================
# DRIVER CODE – DISPLAY LIME EXPLANATIONS
# ==========================================

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(selected_image)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(image_A)
plt.title("LIME - 50 Superpixels")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(image_B)
plt.title("LIME - 150 Superpixels")
plt.axis("off")

plt.show()


# ==========================================
# MAIN LOGIC – CALCULATE IoU
# ==========================================

# Convert masks into binary masks
binary_A = (mask_A > 0)
binary_B = (mask_B > 0)

# Intersection
intersection = np.logical_and(binary_A, binary_B).sum()

# Union
union = np.logical_or(binary_A, binary_B).sum()

# IoU score
iou = intersection / (union + 1e-8)

print("Intersection:", intersection)
print("Union:", union)
print("IoU:", iou)


# ==========================================
# MAIN LOGIC – SELECT IMPORTANT SUPERPIXELS
# ==========================================

def get_important_region_mask(explanation, class_index):

    # Get superpixel weights
    feature_weights = explanation.local_exp[class_index]

    # Select positive weights
    positive_features = [
        item for item in feature_weights if item[1] > 0
    ]

    # Sort by importance
    positive_features.sort(
        key=lambda x: x[1],
        reverse=True
    )

    # Select top 10 regions
    important_segments = [
        item[0] for item in positive_features[:10]
    ]

    # Create binary mask
    segments = explanation.segments

    region_mask = np.isin(
        segments,
        important_segments
    )

    return region_mask


# ==========================================
# MAIN LOGIC – DELETION TEST
# ==========================================

def deletion_test(image, explanation, class_index):

    # Get important regions
    region_mask = get_important_region_mask(
        explanation,
        class_index
    )

    # Copy original image
    deleted_image = image.copy()

    # Remove important regions by replacing with black
    deleted_image[region_mask] = 0

    # Predict after deletion
    new_prediction = predict_images([deleted_image])[0]

    new_probability = new_prediction[class_index]

    return deleted_image, new_probability


# ==========================================
# DRIVER CODE – VALIDATE BOTH EXPLANATIONS
# ==========================================

P0 = original_prediction[target_class]

deleted_A, PA = deletion_test(
    selected_image,
    explanation_A,
    target_class
)

deleted_B, PB = deletion_test(
    selected_image,
    explanation_B,
    target_class
)

# Calculate probability drops
DA = P0 - PA
DB = P0 - PB

print("Original probability:", P0)

print("After deleting A regions:", PA)
print("After deleting B regions:", PB)

print("Probability drop A:", DA)
print("Probability drop B:", DB)

print("Percentage drop A:", (DA / P0) * 100)
print("Percentage drop B:", (DB / P0) * 100)


# ==========================================
# DRIVER CODE – DISPLAY DELETION RESULTS
# ==========================================

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(selected_image)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(deleted_A)
plt.title("After Deleting A")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(deleted_B)
plt.title("After Deleting B")
plt.axis("off")

plt.show()