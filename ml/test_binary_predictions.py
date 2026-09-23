import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from preprocessing import load_and_preprocess_image


EVENT_NAMES = ("fire", "violence", "accident")
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "trained_models"


def load_binary_models(model_dir=DEFAULT_MODEL_DIR):
    """Load the three independently trained binary Keras models."""
    model_dir = Path(model_dir)
    return {
        event_name: tf.keras.models.load_model(
            model_dir / f"binary_{event_name}_model.keras",
            compile=False,
        )
        for event_name in EVENT_NAMES
    }


def predict_binary_image(image_path, model_dir=DEFAULT_MODEL_DIR, threshold=0.5):
    """Return all binary probabilities and the previous highest-event decision."""
    models = load_binary_models(model_dir)
    image = np.expand_dims(load_and_preprocess_image(image_path), axis=0)
    probabilities = {
        event_name: float(model.predict(image, verbose=0).reshape(-1)[0])
        for event_name, model in models.items()
    }
    detected_events = {
        event_name: probability
        for event_name, probability in probabilities.items()
        if probability >= threshold
    }
    event = max(detected_events, key=detected_events.get) if detected_events else "normal"
    return {**probabilities, "event": event}


def main():
    parser = argparse.ArgumentParser(description="Test the independent binary surveillance models.")
    parser.add_argument("image_path", type=Path)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    print(json.dumps(predict_binary_image(args.image_path, args.model_dir, args.threshold), indent=4))


if __name__ == "__main__":
    main()
