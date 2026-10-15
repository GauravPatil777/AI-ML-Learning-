import io
import os

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = FastAPI(title="Cat vs Dog Classifier")

MODEL_PATH = "cat_dog_finetune.keras"

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model file not found: {MODEL_PATH}")



model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={"preprocess_input": preprocess_input}
)

print("Model loaded successfully!")
print("Model input shape:", model.input_shape)
print("Model output shape:", model.output_shape)


@app.get("/")
def home():
    return {"message": "Cat vs Dog classifier is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(
            status_code=415,
            detail="Please upload a JPEG, PNG, or WebP image."
        )

    try:
        contents = await file.read()

        image = Image.open(io.BytesIO(contents)).convert("RGB")
        # IMPORTANT:
        # Replace (224, 224) with the image size used by your notebook
        # if your model expects a different input size.
        image = image.resize((224, 224))

        image = np.array(image, dtype=np.float32)

        # Add batch dimension: changes shape from (28, 28, 1) to (1, 28, 28, 1)
        image = np.expand_dims(image, axis=0)

        # Apply the exact preprocessing used during training.
        # If your notebook uses rescale=1./255 or preprocess_input,
        # reproduce that preprocessing here.

        prediction = model.predict(image, verbose=0)[0][0]
        if prediction >= 0.5:
            label = "dog"
        else:
            label = "cat"
        return {
            "raw_prediction": label,
            "message": (
                "Model inference completed. Interpret the output using "
                "the output activation and class mapping from your notebook."
            )
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not process image: {str(exc)}"
        )