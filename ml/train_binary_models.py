import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from preprocessing import load_and_preprocess_image, SUPPORTED_EXTENSIONS


EVENT_NAMES = ("fire", "violence", "accident")
IMAGE_SHAPE = (128, 128, 3)
DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 32
DEFAULT_SEED = 42
DEFAULT_DATASET_DIR = Path(__file__).resolve().parent / "datasets"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "trained_models"


def build_binary_model():
    """Build the shared binary CNN architecture."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=IMAGE_SHAPE),
            tf.keras.layers.Conv2D(16, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_binary_dataset(event_name, dataset_dir, report=True):
    """Build one binary dataset from event=1 and normal=0 folders."""
    dataset_dir = Path(dataset_dir)
    images = []
    labels = []
    class_counts = {"normal": 0, event_name: 0}
    skipped = 0

    for class_name, label in (("normal", 0), (event_name, 1)):
        class_dir = dataset_dir / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing dataset folder: {class_dir}")

        for image_path in sorted(class_dir.iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                skipped += 1
                continue
            try:
                images.append(load_and_preprocess_image(image_path))
                labels.append(label)
                class_counts[class_name] += 1
            except (OSError, ValueError):
                skipped += 1

    if report:
        print(f"class counts ({event_name} binary): {class_counts}")
        print(f"skipped files: {skipped}")

    height, width = 128, 128
    X = np.stack(images).astype(np.float32) if images else np.empty((0, height, width, 3), dtype=np.float32)
    y = np.asarray(labels, dtype=np.float32)
    return X, y


def split_dataset(images, labels, validation_fraction=0.2, seed=DEFAULT_SEED):
    """Match the multiclass pipeline's reproducible shuffled 80/20 split."""
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


def train_binary_model(
    event_name,
    dataset_dir=DEFAULT_DATASET_DIR,
    output_dir=DEFAULT_OUTPUT_DIR,
    epochs=DEFAULT_EPOCHS,
    batch_size=DEFAULT_BATCH_SIZE,
    seed=DEFAULT_SEED,
):
    """Train and save one event-versus-normal binary classifier."""
    images, labels = load_binary_dataset(event_name, dataset_dir)
    if len(images) < 2:
        raise ValueError(f"Dataset for '{event_name}' does not contain enough images")

    train_images, train_labels, validation_images, validation_labels = split_dataset(
        images, labels, seed=seed
    )

    tf.keras.utils.set_random_seed(seed)
    model = build_binary_model()
    history = model.fit(
        train_images,
        train_labels,
        validation_data=(validation_images, validation_labels),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
    )

    output_path = Path(output_dir) / f"binary_{event_name}_model.keras"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)

    print(f"event: {event_name}")
    print(f"training samples: {len(train_images)}")
    print(f"validation samples: {len(validation_images)}")
    print(f"final training accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"best validation accuracy: {max(history.history['val_accuracy']):.4f}")
    print(f"saved model: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train independent event-versus-normal CNNs.")
    parser.add_argument(
        "--model",
        choices=("all", *EVENT_NAMES),
        default="all",
        help="Train all binary models or one selected event model",
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main():
    args = parse_args()
    selected_events = EVENT_NAMES if args.model == "all" else (args.model,)
    for event_name in selected_events:
        train_binary_model(
            event_name=event_name,
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
