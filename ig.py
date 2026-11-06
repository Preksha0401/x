# ============================================================
# DRIVER CODE - STEP 1: IMPORT LIBRARIES
# ============================================================

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageFilter

from torchvision.models import (
    resnet18,
    ResNet18_Weights
)


# ============================================================
# DRIVER CODE - STEP 2: SELECT DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ============================================================
# MAIN LOGIC - STEP 3: LOAD PRETRAINED RESNET18
# ============================================================

@st.cache_resource
def load_model():

    weights = ResNet18_Weights.DEFAULT

    model = resnet18(weights=weights)

    model = model.to(device)

    model.eval()

    return model, weights


model, weights = load_model()

preprocess = weights.transforms()

categories = weights.meta["categories"]


# ============================================================
# MAIN LOGIC - STEP 4: CREATE BASELINE
# ============================================================

def create_baseline(image, baseline_type):

    image = image.convert("RGB")
    image = image.resize((224, 224))

    if baseline_type == "Black Image":

        return Image.new(
            "RGB", (224, 224), (0, 0, 0)
        )

    elif baseline_type == "White Image":

        return Image.new(
            "RGB", (224, 224), (255, 255, 255)
        )

    elif baseline_type == "Gray Image":

        return Image.new(
            "RGB", (224, 224), (128, 128, 128)
        )

    elif baseline_type == "Blurred Image":

        return image.filter(
            ImageFilter.GaussianBlur(radius=20)
        )

    elif baseline_type == "Gaussian Noise":

        rng = np.random.default_rng(42)

        image_array = np.array(image).astype(np.float32)

        noise = rng.normal(
            loc=128,
            scale=50,
            size=image_array.shape
        )

        noise = np.clip(
            noise, 0, 255
        ).astype(np.uint8)

        return Image.fromarray(noise, mode="RGB")


# ============================================================
# MAIN LOGIC - STEP 5: CONVERT IMAGE TO TENSOR
# ============================================================

def image_to_tensor(image):

    tensor = preprocess(image)

    tensor = tensor.unsqueeze(0)

    return tensor.to(device)


# ============================================================
# MAIN LOGIC - STEP 6: MODEL PREDICTION
# ============================================================

def predict(image):

    tensor = image_to_tensor(image)

    with torch.no_grad():

        output = model(tensor)

        probabilities = torch.softmax(
            output, dim=1
        )

        top_probabilities, top_indices = torch.topk(
            probabilities, 5
        )

    return (
        top_probabilities[0].cpu().numpy(),
        top_indices[0].cpu().numpy(),
        output
    )


# ============================================================
# MAIN LOGIC - STEP 7: MANUAL INTEGRATED GRADIENTS
# ============================================================

def manual_integrated_gradients(
    input_tensor,
    baseline_tensor,
    target_class,
    n_steps=50
):

    input_tensor = input_tensor.float()
    baseline_tensor = baseline_tensor.float()

    # Difference between input and baseline
    difference = input_tensor - baseline_tensor

    # Store accumulated gradients
    accumulated_gradients = torch.zeros_like(
        input_tensor
    )

    # Numerical integration
    for step in range(n_steps + 1):

        alpha = step / n_steps

        # Generate intermediate input
        interpolated = (
            baseline_tensor + alpha * difference
        )

        interpolated = (
            interpolated.clone().detach()
        )

        interpolated.requires_grad_(True)

        model.zero_grad(set_to_none=True)

        # Forward pass
        output = model(interpolated)

        target_output = output[0, target_class]

        # Backward pass
        target_output.backward()

        gradient = interpolated.grad.detach()

        # Trapezoidal rule
        if step == 0 or step == n_steps:

            accumulated_gradients += 0.5 * gradient

        else:

            accumulated_gradients += gradient

    # Average gradient
    average_gradient = (
        accumulated_gradients / n_steps
    )

    # Integrated Gradients
    attributions = difference * average_gradient

    # ========================================================
    # CONVERGENCE DELTA
    # ========================================================

    with torch.no_grad():

        input_output = model(
            input_tensor
        )[0, target_class]

        baseline_output = model(
            baseline_tensor
        )[0, target_class]

    attribution_sum = attributions.sum()

    output_difference = (
        input_output - baseline_output
    )

    convergence_delta = (
        attribution_sum - output_difference
    )

    return (
        attributions.detach(),
        convergence_delta.detach()
    )


# ============================================================
# MAIN LOGIC - STEP 8: PROCESS ATTRIBUTIONS
# ============================================================

def process_attributions(attributions):

    attr = (
        attributions
        .squeeze(0)
        .detach()
        .cpu()
        .numpy()
    )

    # Convert C,H,W to H,W,C
    attr = np.transpose(attr, (1, 2, 0))

    # Combine RGB channels
    signed_map = np.sum(attr, axis=2)

    positive_map = np.maximum(signed_map, 0)

    negative_map = np.minimum(signed_map, 0)

    absolute_map = np.abs(signed_map)

    return (
        signed_map,
        positive_map,
        negative_map,
        absolute_map
    )


# ============================================================
# MAIN LOGIC - STEP 9: CALCULATE ATTRIBUTION SCORES
# ============================================================

def calculate_scores(attributions):

    raw = attributions.detach().cpu().numpy()

    positive = np.sum(np.maximum(raw, 0))

    negative = np.sum(np.minimum(raw, 0))

    absolute = np.sum(np.abs(raw))

    if absolute > 0:

        positive_percentage = (
            positive / absolute * 100
        )

        negative_percentage = (
            abs(negative) / absolute * 100
        )

    else:

        positive_percentage = 0
        negative_percentage = 0

    return {
        "positive": positive,
        "negative": negative,
        "absolute": absolute,
        "positive_percentage": positive_percentage,
        "negative_percentage": negative_percentage
    }


