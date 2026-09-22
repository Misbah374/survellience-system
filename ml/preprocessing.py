from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError


IMAGE_SIZE = (128, 128)
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


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


def _find_split_folder(dataset_path, split_names):
	for split_name in split_names:
		folder = dataset_path / split_name
		if folder.is_dir():
			return folder
	return None


def load_dataset(dataset_path, image_size=IMAGE_SIZE, report=True):
	"""Load positive and negative images from a dataset directory.

	Returns:
		X: Float32 image arrays with shape (n, height, width, 3).
		y: Integer labels where positive is 1 and negative is 0.
	"""
	dataset_path = Path(dataset_path)
	split_folders = {
		1: _find_split_folder(dataset_path, ("positive", "positives")),
		0: _find_split_folder(dataset_path, ("negative", "negatives")),
	}
	images = []
	labels = []
	skipped = 0
	loaded_counts = {1: 0, 0: 0}

	for label, folder in split_folders.items():
		if folder is None:
			continue
		for image_path in sorted(folder.iterdir()):
			if not image_path.is_file() or image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
				skipped += 1
				continue
			try:
				images.append(load_and_preprocess_image(image_path, image_size))
				labels.append(label)
				loaded_counts[label] += 1
			except (OSError, UnidentifiedImageError, ValueError):
				skipped += 1

	height, width = image_size[1], image_size[0]
	X = np.stack(images).astype(np.float32) if images else np.empty((0, height, width, 3), dtype=np.float32)
	y = np.asarray(labels, dtype=np.int64)

	if report:
		label_distribution = {
			int(label): int(count)
			for label, count in zip(*np.unique(y, return_counts=True))
		}
		print(f"positive images: {loaded_counts[1]}")
		print(f"negative images: {loaded_counts[0]}")
		print(f"image shape: {X.shape[1:]}")
		print(f"label distribution: {label_distribution}")
		print(f"skipped files: {skipped}")

	return X, y
