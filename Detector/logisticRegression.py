import kagglehub
import cv2
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

path = Path(kagglehub.dataset_download("birdy654/cifake-real-and-ai-generated-synthetic-images"))
CACHE = Path(__file__).parent / "cache"
FEATURES = "pixels_fft" # part of the cache filename, change it whenever read_features changes

# an image is just a matrix of pixel values.
# 32x32 RGB -> flatten into a 3072-length vector -> one "row" for the model.
# a linear model can't find texture artifacts in raw pixels, so we also add the
# log-magnitude FFT of the grayscale image (1024 values): generated images tend
# to have unusual high-frequency patterns that show up in the spectrum.
def read_features(fname):
    img = cv2.imread(str(fname))
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    spectrum = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(gray))))
    return np.hstack([img.flatten(), spectrum.flatten()]).astype(np.float32)


def load_split(split_dir):
    X, y = [], []
    for label, folder in [(1, "FAKE"), (0, "REAL")]:
        files = list((split_dir / folder).iterdir())
        with ThreadPoolExecutor() as pool: # imread is I/O + JPEG decode, threads help a lot
            for img in pool.map(read_features, files):
                if img is not None:
                    X.append(img)
                    y.append(label)
    return np.array(X), np.array(y)


def get_split(name):
    cache_x, cache_y = CACHE / f"X_{name}_{FEATURES}.npy", CACHE / f"y_{name}_{FEATURES}.npy"
    if cache_x.exists():
        return np.load(cache_x), np.load(cache_y)
    X, y = load_split(path / name)
    CACHE.mkdir(exist_ok=True)
    np.save(cache_x, X)
    np.save(cache_y, y)
    return X, y


X_train, y_train = get_split("train")
X_test, y_test = get_split("test")

print(f"train: {X_train.shape}, test: {X_test.shape}", flush=True)

# SGDClassifier with log_loss is logistic regression trained via SGD.
# lbfgs would need many full passes over 100k images; SGD converges in a handful of epochs.
# SGD is sensitive to feature scale, and pixels (0-255) and FFT magnitudes live on very
# different scales, so StandardScaler gives every feature mean 0 / std 1 first.
# inside the pipeline the scaler only learns its mean/std from the training data.
model = make_pipeline(
    StandardScaler(),
    SGDClassifier(loss="log_loss", max_iter=1000, tol=1e-5, alpha=1e-5, average=True, random_state=42),
)
model.fit(X_train, y_train)

preds = model.predict(X_test)
print(f"accuracy: {accuracy_score(y_test, preds):.4f}")
print(confusion_matrix(y_test, preds))
print(classification_report(y_test, preds, target_names=["REAL", "FAKE"]))
