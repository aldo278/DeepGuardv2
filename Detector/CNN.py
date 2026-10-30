import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from pathlib import Path
import kagglehub


path = Path(kagglehub.dataset_download("birdy654/cifake-real-and-ai-generated-synthetic-images"))

(train_images, train_labels), (test_images, test_labels) = path
