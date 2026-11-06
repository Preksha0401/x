# ============================================================
# DRIVER CODE - STEP 1: IMPORT LIBRARIES
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from PIL import Image, ImageDraw

import saliency.core as saliency


# ============================================================
# DRIVER CODE - STEP 2: SELECT DEVICE
# ============================================================

print("TensorFlow version:", tf.__version__)


# ============================================================
# MAIN LOGIC - STEP 3: LOAD PRETRAINED INCEPTIONV3
# ============================================================

model = tf.keras.applications.InceptionV3(
    weights="imagenet",
    include_top=True
)

model.trainable = False

print("InceptionV3 loaded successfully!")


# ============================================================
# DRIVER CODE - STEP 4: LOAD IMAGE
# ============================================================

image = Image.open("dog.jpg").convert("RGB")

image = image.resize((299, 299))

input_image = np.array(image).astype(np.float32)

plt.imshow(image)
plt.title("Original Image")
plt.axis("off")
plt.show()


# ============================================================
# MAIN LOGIC - STEP 5: MODEL PREDICTION
# ============================================================

def predict(image_array):

    image_batch = np.expand_dims(
        image_array, axis=0
    )

    processed = tf.keras.applications.inception_v3.preprocess_input(
        image_batch.copy()
    )

    predictions = model.predict(
        processed, verbose=0
    )

    return predictions[0]


predictions = predict(input_image)

target_class = np.argmax(predictions)

confidence = predictions[target_class] * 100

print("Predicted class index:", target_class)
print("Confidence:", confidence, "%")


# ============================================================
# MAIN LOGIC - STEP 6: DEFINE MODEL GRADIENT FUNCTION
# ============================================================

def call_model_function(
    images,
    call_model_args=None,
    expected_keys=None
):

    images = tf.convert_to_tensor(
        images, dtype=tf.float32
    )

    with tf.GradientTape() as tape:

        tape.watch(images)

        processed = tf.keras.applications.inception_v3.preprocess_input(
            images
        )

        outputs = model(processed, training=False)

        target_output = outputs[
            :, call_model_args["class_index"]
        ]

    gradients = tape.gradient(
        target_output, images
    )

    result = {}

    if saliency.base.INPUT_OUTPUT in expected_keys:

        result[saliency.base.INPUT_OUTPUT] = (
            target_output.numpy()
        )

    if saliency.base.INPUT_GRADIENTS in expected_keys:

        result[saliency.base.INPUT_GRADIENTS] = (
            gradients.numpy()
        )

    return result


# ============================================================
# MAIN LOGIC - STEP 7: CREATE XRAI OBJECT
# ============================================================

xrai_object = saliency.XRAI()


# ============================================================
# MAIN LOGIC - STEP 8: COMPUTE XRAI ATTRIBUTION MAP
# ============================================================

xrai_attributions = xrai_object.GetMask(
    input_image,
    call_model_function,
    call_model_args={
        "class_index": target_class
    }
)

print("XRAI attribution generated!")


# ============================================================
# MAIN LOGIC - STEP 9: NORMALIZE XRAI HEATMAP
# ============================================================

xrai_normalized = (
    xrai_attributions -
    np.min(xrai_attributions)
)

xrai_normalized = (
    xrai_normalized /
    (np.max(xrai_normalized) + 1e-8)
)


# ============================================================
# DRIVER CODE - STEP 10: VISUALIZE XRAI HEATMAP
# ============================================================

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)

plt.imshow(image)

plt.title("Original Image")

plt.axis("off")


plt.subplot(1, 2, 2)

plt.imshow(
    input_image.astype(np.uint8)
)

plt.imshow(
    xrai_normalized,
    cmap="jet",
    alpha=0.5
)

plt.title("XRAI Heatmap")

plt.axis("off")

plt.show()


# ============================================================
# MAIN LOGIC - STEP 11: DISPLAY TOP 20% REGIONS
# ============================================================

threshold = np.percentile(
    xrai_normalized, 80
)

top_mask = (
    xrai_normalized >= threshold
)

top_regions = (
    input_image * top_mask[:, :, np.newaxis]
).astype(np.uint8)

plt.figure(figsize=(6, 5))

plt.imshow(top_regions)

plt.title("Top 20% XRAI Regions")

plt.axis("off")

plt.show()


# ============================================================
# OCCLUSION ANALYSIS
# ============================================================

# ============================================================
# MAIN LOGIC - STEP 12: CREATE GRAY PATCH
# ============================================================

def create_gray_patch(
    image,
    x,
    y,
    patch_size
):

    patched_image = image.copy()

    draw = ImageDraw.Draw(patched_image)

    draw.rectangle(
        [
            x,
            y,
            x + patch_size,
            y + patch_size
        ],
        fill=(128, 128, 128)
    )

    return patched_image


# ============================================================
# MAIN LOGIC - STEP 13: GET MODEL CONFIDENCE
# ============================================================

def get_confidence(image):

    image_array = np.array(
        image
    ).astype(np.float32)

    probabilities = predict(image_array)

    return probabilities[target_class] * 100


# ============================================================
# MAIN LOGIC - STEP 14: ORIGINAL CONFIDENCE
# ============================================================

original_confidence = get_confidence(image)

print(
    "Original Confidence:",
    original_confidence
)


# ============================================================
# MAIN LOGIC - STEP 15: TEST DIFFERENT PATCH SIZES
# ============================================================

patch_sizes = [10, 30, 50]

for patch_size in patch_sizes:

    best_drop = -float("inf")

    best_position = None

    # Move patch across image
    for y in range(
        0, 299 - patch_size, 10
    ):

        for x in range(
            0, 299 - patch_size, 10
        ):

            patched_image = create_gray_patch(
                image,
                x,
                y,
                patch_size
            )

            patched_confidence = get_confidence(
                patched_image
            )

            confidence_drop = (
                original_confidence -
                patched_confidence
            )

            if confidence_drop > best_drop:

                best_drop = confidence_drop

                best_position = (x, y)

    print("\nPatch Size:", patch_size)

    print("Best Position:", best_position)

    print("Confidence Drop:", best_drop)


# ============================================================
# DRIVER CODE - STEP 16: DISPLAY FINAL PATCH
# ============================================================

x, y = best_position

final_patch = create_gray_patch(
    image,
    x,
    y,
    patch_sizes[-1]
)

plt.imshow(final_patch)

plt.title("Gray Patch Occlusion")

plt.axis("off")

plt.show()