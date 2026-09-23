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
EVENT_NAMES = ("fire", "violence", "accident")
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "trained_models" / "surveillance_model.keras"


def load_model_file(model_path):
	"""Load the multiclass Keras model."""
	return tf.keras.models.load_model(Path(model_path), compile=False)


def load_models(model_directory=None):
	"""Load the multiclass and binary models used for comparison."""
	model_directory = (
		DEFAULT_MODEL_PATH.parent
		if model_directory is None
		else Path(model_directory)
	)
	if not model_directory.is_dir():
		model_directory = model_directory.parent
	return {
		"multiclass": load_model_file(model_directory / "surveillance_model.keras"),
		"binary": {
			event_name: tf.keras.models.load_model(
				model_directory / f"binary_{event_name}_model.keras",
				compile=False,
			)
			for event_name in EVENT_NAMES
		},
	}


def _predict_with_model(image, model):
	"""Run the multiclass model on one preprocessed image."""
	model_input = np.expand_dims(image, axis=0)
	probabilities = model.predict(model_input, verbose=0).reshape(-1)
	return {
		**{name: float(probability) for name, probability in zip(CLASS_NAMES, probabilities)},
		"event": CLASS_NAMES[int(np.argmax(probabilities))],
	}


def _predict_with_models(image, models, threshold=0.5):
	"""Run both model approaches on one already-preprocessed image."""
	multiclass_result = _predict_with_model(image, models["multiclass"])
	model_input = np.expand_dims(image, axis=0)
	binary_probabilities = {
		event_name: float(
			models["binary"][event_name].predict(model_input, verbose=0).reshape(-1)[0]
		)
		for event_name in EVENT_NAMES
	}
	detected_events = {
		event_name: probability
		for event_name, probability in binary_probabilities.items()
		if probability >= threshold
	}
	binary_event = max(detected_events, key=detected_events.get) if detected_events else "normal"
	binary_result = {**binary_probabilities, "event": binary_event}
	return {
		"multiclass": multiclass_result,
		"binary": binary_result,
		"comparison": {
			"multiclass_event": multiclass_result["event"],
			"binary_event": binary_event,
			"agreement": multiclass_result["event"] == binary_event,
		},
	}


def predict_rgb_array(image, models=None, threshold=0.5):
	"""Run both approaches on one RGB frame using shared preprocessing."""
	if models is None:
		models = load_models()
	processed_image = preprocess_image_array(image)
	return _predict_with_models(processed_image, models, threshold)


def predict_image(image_path, model_directory=None):
	"""Run both approaches on one image and return the combined result."""
	models = load_models(model_directory)
	image = load_and_preprocess_image(image_path)
	return _predict_with_models(image, models)


def main():
	parser = argparse.ArgumentParser(description="Run surveillance event detection on one image.")
	parser.add_argument("image_path", type=Path, help="Path to the image to analyze")
	args = parser.parse_args()
	result = predict_image(args.image_path)
	print(json.dumps(result, indent=4))


if __name__ == "__main__":
	main()
