import kagglehub
import cv2
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

path = Path(kagglehub.dataset_download("birdy654/cifake-real-and-ai-generated-synthetic-images"))
CACHE = Path(__file__).parent / "cache"

# an image is just a matrix of pixel values.
# 32x32 RGB -> flatten into a 3072-length vector -> one "row" for the model
def read_flat(fname):
    img = cv2.imread(str(fname))
    return None if img is None else img.flatten()


def load_split(split_dir):
    X, y = [], []
    for label, folder in [(1, "FAKE"), (0, "REAL")]:
        files = list((split_dir / folder).iterdir())
        with ThreadPoolExecutor() as pool: # imread is I/O + JPEG decode, threads help a lot
            for img in pool.map(read_flat, files):
                if img is not None:
                    X.append(img)
                    y.append(label)
    return np.array(X), np.array(y)


def get_split(name):
    cache_x, cache_y = CACHE / f"X_{name}.npy", CACHE / f"y_{name}.npy"
    if cache_x.exists():
        return np.load(cache_x), np.load(cache_y)
    X, y = load_split(path / name)
    CACHE.mkdir(exist_ok=True)
    np.save(cache_x, X)
    np.save(cache_y, y)
    return X, y


X_train, y_train = get_split("train")
X_test, y_test = get_split("test")

X_train = X_train.astype(np.float32) / 255.0 # scale pixels 0-255 -> 0-1
X_test = X_test.astype(np.float32) / 255.0

print(f"train: {X_train.shape}, test: {X_test.shape}", flush=True)

# SGDClassifier with log_loss is logistic regression trained via SGD.
# lbfgs does 1000 full passes over 100k images; SGD converges in a handful of epochs
model = SGDClassifier(loss="log_loss", max_iter=1000, tol=1e-5, alpha=1e-5, average=True, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)
print(f"accuracy: {accuracy_score(y_test, preds):.4f}")
print(confusion_matrix(y_test, preds))
print(classification_report(y_test, preds, target_names=["REAL", "FAKE"]))
