from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError


IMAGE_SIZE = (128, 128)
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
CLASS_LABELS = {
	"normal": 0,
	"fire": 1,
	"accident": 2,
	"violence": 3,
}


def load_and_preprocess_image(image_path, image_size=IMAGE_SIZE):
	"""Load one image and return a normalized RGB array for CNN input."""
	with Image.open(image_path) as image:
		image = image.convert("RGB")
		image = image.resize(image_size, Image.Resampling.LANCZOS)
		return np.asarray(image, dtype=np.float32) / 255.0


def preprocess_image_array(image, image_size=IMAGE_SIZE):
	"""Preprocess an RGB image array using the same pipeline as image files."""
	image = Image.fromarray(image).convert("RGB")
	image = image.resize(image_size, Image.Resampling.LANCZOS)
	return np.asarray(image, dtype=np.float32) / 255.0


def load_dataset(dataset_path, image_size=IMAGE_SIZE, report=True):
	"""Load the four class folders from a dataset directory.

	Returns:
		X: Float32 image arrays with shape (n, height, width, 3).
		y: Integer class labels for normal, fire, accident, and violence.
	"""
	dataset_path = Path(dataset_path)
	images = []
	labels = []
	skipped = 0
	loaded_counts = {class_name: 0 for class_name in CLASS_LABELS}

	for class_name, label in CLASS_LABELS.items():
		folder = dataset_path / class_name
		if not folder.is_dir():
			continue
		for image_path in sorted(folder.iterdir()):
			if not image_path.is_file() or image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
				skipped += 1
				continue
			try:
				images.append(load_and_preprocess_image(image_path, image_size))
				labels.append(label)
				loaded_counts[class_name] += 1
			except (OSError, UnidentifiedImageError, ValueError):
				skipped += 1

	height, width = image_size[1], image_size[0]
	X = np.stack(images).astype(np.float32) if images else np.empty((0, height, width, 3), dtype=np.float32)
	y = np.asarray(labels, dtype=np.int64)

	if report:
		print(f"class counts: {loaded_counts}")
		print(f"image shape: {X.shape[1:]}")
		print(f"label distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
		print(f"skipped files: {skipped}")

	return X, y
