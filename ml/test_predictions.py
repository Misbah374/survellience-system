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

    print(f"\nImage: {image_path}")

    probabilities = model.predict(image, verbose=0)[0]
    for name, probability in zip(CLASS_NAMES, probabilities):
        print(f"{name:10s}: {float(probability):.4f}")


# CHANGE THESE TWO PATHS
test_image(r"D:\coding_projects\5thsemproject\test_images\accident.jpg")
test_image(r"D:\coding_projects\5thsemproject\test_images\non-accident.jpg")
test_image(r"D:\coding_projects\5thsemproject\test_images\normal1.jpg")
test_image(r"D:\coding_projects\5thsemproject\test_images\normal2.png")
test_image(r"D:\coding_projects\5thsemproject\test_images\normal3.jpg")
test_image(r"D:\coding_projects\5thsemproject\test_images\normal4.png")
test_image(r"D:\coding_projects\5thsemproject\test_images\normal5.jpg")