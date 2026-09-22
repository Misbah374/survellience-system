import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import tensorflow as tf

try:
	from .preprocessing import load_and_preprocess_image, preprocess_image_array
except ImportError:
	from preprocessing import load_and_preprocess_image, preprocess_image_array


IMAGE_SIZE = (128, 128, 3)
DATASET_NAMES = ("fire", "violence", "accident")


def load_model_file(model_path):
	"""Load one required .pt artifact through a temporary HDF5 filename."""
	model_path = Path(model_path)
	with TemporaryDirectory() as temporary_directory:
		temporary_path = Path(temporary_directory) / "model.h5"
		temporary_path.write_bytes(model_path.read_bytes())
		return tf.keras.models.load_model(temporary_path, compile=False)


def load_models(model_directory=None):
	"""Load and return all event models before any prediction is made."""
	if model_directory is None:
		model_directory = Path(__file__).resolve().parent / "models"
	model_directory = Path(model_directory)
	return {
		dataset_name: load_model_file(
			model_directory / f"{dataset_name}_model.pt"
		)
		for dataset_name in DATASET_NAMES
	}


def _predict_with_models(image, models, threshold=0.5):
	"""Run already-loaded models on one preprocessed image."""
	model_input = np.expand_dims(image, axis=0)
	probabilities = {
		dataset_name: float(
			models[dataset_name].predict(model_input, verbose=0).reshape(-1)[0]
		)
		for dataset_name in DATASET_NAMES
	}
	detections = {
		dataset_name: {
			"probability": probability,
			"detected": probability >= threshold,
		}
		for dataset_name, probability in probabilities.items()
	}
	detected_events = {
		dataset_name: probability
		for dataset_name, probability in probabilities.items()
		if probability >= threshold
	}
	event = max(detected_events, key=detected_events.get) if detected_events else "normal"
	return {**detections, "event": event}


def predict_rgb_array(image, models=None, threshold=0.5):
	"""Run detection on one RGB frame array using the shared preprocessing pipeline."""
	if models is None:
		models = load_models()
	processed_image = preprocess_image_array(image)
	return _predict_with_models(processed_image, models, threshold)


def predict_image(image_path, threshold=0.5, model_directory=None):
	"""Run all event models on one image and return the detection result."""
	models = load_models(model_directory)
	image = load_and_preprocess_image(image_path)
	return _predict_with_models(image, models, threshold)


def main():
	parser = argparse.ArgumentParser(description="Run surveillance event detection on one image.")
	parser.add_argument("image_path", type=Path, help="Path to the image to analyze")
	args = parser.parse_args()
	result = predict_image(args.image_path)
	print(json.dumps(result, indent=4))


if __name__ == "__main__":
	main()
