import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from preprocessing import load_dataset


CLASS_NAMES = ("normal", "fire", "accident", "violence")
IMAGE_SHAPE = (128, 128, 3)
DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 32
DEFAULT_SEED = 42
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "trained_models"
DEFAULT_OUTPUT_PATH = DEFAULT_OUTPUT_DIR / "surveillance_model.keras"


def build_model():
    """Build the four-class surveillance CNN."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=IMAGE_SHAPE),
            tf.keras.layers.Conv2D(16, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(4, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


def split_dataset(images, labels, validation_fraction=0.2, seed=DEFAULT_SEED):
    """Return a reproducible shuffled 80/20 train/validation split."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(images))
    validation_size = int(len(indices) * validation_fraction)
    validation_indices = indices[:validation_size]
    training_indices = indices[validation_size:]
    return (
        images[training_indices],
        labels[training_indices],
        images[validation_indices],
        labels[validation_indices],
    )


def train_model(
    dataset_dir,
    output_path,
    epochs=DEFAULT_EPOCHS,
    batch_size=DEFAULT_BATCH_SIZE,
    seed=DEFAULT_SEED,
):
    """Load, train, report, and save the multiclass classifier."""
    dataset_path = Path(dataset_dir)
    output_path = Path(output_path)

    images, labels = load_dataset(dataset_path, report=True)
    if len(images) < 2:
        raise ValueError("Dataset does not contain enough images")

    train_images, train_labels, validation_images, validation_labels = split_dataset(
        images, labels, seed=seed
    )

    tf.keras.utils.set_random_seed(seed)
    model = build_model()
    history = model.fit(
        train_images,
        train_labels,
        validation_data=(validation_images, validation_labels),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)

    print(f"classes: {CLASS_NAMES}")
    print(f"training samples: {len(train_images)}")
    print(f"validation samples: {len(validation_images)}")
    print(f"final training accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"saved model: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train the multiclass surveillance CNN.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "datasets",
        help="Directory containing the four class folders",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path for the new Keras model file",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main():
    args = parse_args()
    train_model(
        dataset_dir=args.dataset_dir,
        output_path=args.output_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
