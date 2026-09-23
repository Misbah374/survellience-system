import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from preprocessing import load_and_preprocess_image


MODEL_PATH = Path(__file__).resolve().parent / "trained_models" / "surveillance_model.keras"
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
CLASS_NAMES = ("normal", "fire", "accident", "violence")


def test_image(image_path):
    image = load_and_preprocess_image(image_path)
    image = np.expand_dims(image, axis=0)
    probabilities = model.predict(image, verbose=0)[0]
    return {
        **{name: float(probability) for name, probability in zip(CLASS_NAMES, probabilities)},
        "event": CLASS_NAMES[int(np.argmax(probabilities))],
    }


def main():
    parser = argparse.ArgumentParser(description="Test the multiclass surveillance model.")
    parser.add_argument("image_path", type=Path)
    args = parser.parse_args()
    print(json.dumps(test_image(args.image_path), indent=4))


if __name__ == "__main__":
    main()