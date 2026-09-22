import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

try:
	from .preprocessing import load_and_preprocess_image, preprocess_image_array
except ImportError:
	from preprocessing import load_and_preprocess_image, preprocess_image_array


IMAGE_SIZE = (128, 128, 3)
CLASS_NAMES = ("normal", "fire", "accident", "violence")
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "trained_models" / "surveillance_model.keras"


def load_model_file(model_path):
	"""Load the multiclass Keras model."""
	return tf.keras.models.load_model(Path(model_path), compile=False)


def load_models(model_directory=None):
	"""Load the single multiclass model for prediction."""
	model_path = DEFAULT_MODEL_PATH if model_directory is None else Path(model_directory)
	if model_path.is_dir():
		model_path = model_path / "surveillance_model.keras"
	return load_model_file(model_path)


def _predict_with_model(image, model):
	"""Run the multiclass model on one preprocessed image."""
	model_input = np.expand_dims(image, axis=0)
	probabilities = model.predict(model_input, verbose=0).reshape(-1)
	return {
		**{name: float(probability) for name, probability in zip(CLASS_NAMES, probabilities)},
		"event": CLASS_NAMES[int(np.argmax(probabilities))],
	}


def predict_rgb_array(image, model=None):
	"""Run detection on one RGB frame array using the shared preprocessing pipeline."""
	if model is None:
		model = load_models()
	processed_image = preprocess_image_array(image)
	return _predict_with_model(processed_image, model)


def predict_image(image_path, model_directory=None):
	"""Run the multiclass model on one image and return the detection result."""
	model = load_models(model_directory)
	image = load_and_preprocess_image(image_path)
	return _predict_with_model(image, model)


def main():
	parser = argparse.ArgumentParser(description="Run surveillance event detection on one image.")
	parser.add_argument("image_path", type=Path, help="Path to the image to analyze")
	args = parser.parse_args()
	result = predict_image(args.image_path)
	print(json.dumps(result, indent=4))


if __name__ == "__main__":
	main()
