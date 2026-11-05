# a computer vision library, used here to decode image bytes and resize them
import cv2

# arrays and math
import numpy as np

import time
from pathlib import Path

# Training uses the 32x32 CIFAKE pixels that logisticRegression.py already extracted into cache/. The first 3072 columns of each cached row are the flattened BGR pixels (32*32*3),
# the other 1024 are the FFT features the logistic regression uses, which the CNN does not need. Labels are 1 = FAKE, 0 = REAL.
CACHE = Path(__file__).parent / "cache"
FEATURES = "pixels_fft"
MODEL_PATH = Path(__file__).parent / "models" / "cnn.keras"

# my own labeled images (full size, any size) live in DeepGuardv2/Data/Real and Data/Fake. Fine-tuning on them is saved to a separate file so the CIFAKE model is never overwritten
DATA_DIR = Path(__file__).parent.parent / "Data"
DATA_MODEL_PATH = Path(__file__).parent / "models" / "cnn_data.keras"

# CIFAKE images are 32x32, so every image is resized to this before it reaches the model. Real photos are shrunk with INTER_AREA, the same kind of downscaling CIFAKE's images went through,
# which keeps the pixel statistics close to what the model saw in training.
SIZE = 32


def load_pixels(name):
    X = np.load(CACHE / f"X_{name}_{FEATURES}.npy", mmap_mode="r")
    y = np.load(CACHE / f"y_{name}_{FEATURES}.npy")
    # slice in chunks so we never hold the whole 1.6 GB feature file as floats, and store the pixels as uint8 (0-255)
    pixels = np.empty((len(X), SIZE, SIZE, 3), dtype=np.uint8)
    for i in range(0, len(X), 10000):
        pixels[i:i + 10000] = X[i:i + 10000, :SIZE * SIZE * 3].reshape(-1, SIZE, SIZE, 3)
    return pixels, y


# The model works for any image size: the input shape is (None, None, 3), where None means "any height / width", and there is no Flatten layer.
# Flatten would force one fixed size, because its output length depends on the image size. Instead, GlobalAveragePooling2D and GlobalMaxPooling2D collapse each feature map to one number
# no matter how big the map is, so the Dense layers after them always get the same number of inputs. predict() still resizes to 32x32 to match the training data, but the network itself
# does not care about the size.
def build_model():
    from tensorflow.keras import layers, models

    inputs = layers.Input(shape=(None, None, 3))
    x = layers.Rescaling(1 / 255)(inputs)  # pixels 0-255 -> 0-1

    # two Conv2D + ReLU, then MaxPooling halves the size. Filters double each block (32 -> 64 -> 128) to find more complex patterns as the image gets smaller
    for filters in (32, 64):
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)

    # average = "how much of this pattern is there overall", max = "is this pattern anywhere". Generated images often have local artifacts, so we keep both
    x = layers.Concatenate()([layers.GlobalAveragePooling2D()(x), layers.GlobalMaxPooling2D()(x)])
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)  # randomly switches off 30% of units while training, which fights overfitting

    # one output between 0 and 1 (sigmoid) = probability the image is FAKE
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train(epochs=12):
    from tensorflow.keras import callbacks

    X_train, y_train = load_pixels("train")
    X_test, y_test = load_pixels("test")
    print(f"train: {X_train.shape}, test: {X_test.shape}", flush=True)

    model = build_model()
    model.summary()

    # 10% of the training data is held back to watch for overfitting. The test set is only used once at the end. Training stops when validation loss stops improving for 3 epochs
    # and goes back to the best epoch
    model.fit(
        X_train, y_train, batch_size=128, epochs=epochs, validation_split=0.1,
        callbacks=[callbacks.EarlyStopping(patience=3, restore_best_weights=True)],
        verbose=2,
    )

    loss, acc = model.evaluate(X_test, y_test, batch_size=256, verbose=0)
    print(f"test accuracy: {acc:.4f}", flush=True)

    MODEL_PATH.parent.mkdir(exist_ok=True)
    model.save(MODEL_PATH)
    print(f"saved model to {MODEL_PATH}", flush=True)
    return model


# the saved model is loaded once and reused for every request
_model = None

def load_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"{MODEL_PATH} not found, run `python CNN.py` to train and save it")
        from tensorflow.keras.models import load_model as keras_load
        _model = keras_load(MODEL_PATH)
    return _model


# any image size in, a (1, 32, 32, 3) BGR batch out. cv2 keeps channels in BGR order, the same as the cached training pixels
def prepare(img):
    return cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)[None].astype(np.float32)


# the shared detector interface: image file bytes in, result dict out (see PLAN.md section 4)
def predict(image_bytes):
    start = time.perf_counter()
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("not a readable image")
    prob_fake = float(load_model()(prepare(img), training=False)[0, 0])
    h, w = img.shape[:2]
    return {
        "model": "cnn",
        "verdict": "FAKE" if prob_fake >= 0.5 else "REAL",
        "prob_fake": prob_fake,
        "detail": f"{w}x{h} image resized to {SIZE}x{SIZE}",
        "ms": round((time.perf_counter() - start) * 1000),
    }


# ---- fine-tuning on my own images (python CNN.py --data) ----

def read_folder(folder):
    imgs = []
    for f in sorted((DATA_DIR / folder).glob("*")):
        img = cv2.imread(str(f))
        if img is not None:
            imgs.append(img)
    return imgs


# Only a few dozen images, so each one is turned into many training samples: random crops covering 40-100% of the image, flipped half the time, then shrunk to 32x32.
# The crops also teach the model that a piece of a photo is still real/fake, which helps with big images of different shapes.
def make_views(imgs, label, n_views, rng):
    X, y = [], []
    for img in imgs:
        h, w = img.shape[:2]
        for _ in range(n_views):
            scale = rng.uniform(0.4, 1.0)
            ch, cw = max(int(h * scale), 8), max(int(w * scale), 8)
            top, left = rng.integers(0, h - ch + 1), rng.integers(0, w - cw + 1)
            view = img[top:top + ch, left:left + cw]
            if rng.random() < 0.5:
                view = view[:, ::-1]
            X.append(cv2.resize(np.ascontiguousarray(view), (SIZE, SIZE), interpolation=cv2.INTER_AREA))
            y.append(label)
    return np.array(X, dtype=np.uint8), np.array(y)


# Starts from the CIFAKE-trained model (training from scratch on ~30 images would learn nothing) and nudges it with a small learning rate. The split is by IMAGE: whole images are held out
# before any crops are made, so a crop of a training image can never show up in the held-out set. The held-out score uses the whole image, exactly like predict() does.
def train_on_data(epochs=15, val_frac=0.25, n_views=40, seed=0):
    from tensorflow.keras import callbacks, optimizers
    from tensorflow.keras.models import load_model as keras_load

    rng = np.random.default_rng(seed)
    classes = {"Real": 0, "Fake": 1}
    train_X, train_y, val_X, val_y = [], [], [], []
    for folder, label in classes.items():
        imgs = read_folder(folder)
        order = rng.permutation(len(imgs))
        n_val = max(1, round(len(imgs) * val_frac))
        held = [imgs[i] for i in order[:n_val]]
        kept = [imgs[i] for i in order[n_val:]]
        print(f"{folder}: {len(kept)} train images, {len(held)} held-out images", flush=True)
        X, y = make_views(kept, label, n_views, rng)
        train_X.append(X); train_y.append(y)
        val_X += [prepare(i)[0] for i in held]; val_y += [label] * len(held)
    X_train, y_train = np.concatenate(train_X), np.concatenate(train_y)
    X_val, y_val = np.array(val_X, dtype=np.float32), np.array(val_y)

    # real and fake image counts differ, so each class gets a weight that makes both count equally
    counts = np.bincount(y_train)
    weights = {c: len(y_train) / (2 * counts[c]) for c in (0, 1)}

    base = MODEL_PATH
    if not base.exists():
        raise FileNotFoundError(f"{base} not found, run `python CNN.py` first to train on CIFAKE")
    model = keras_load(base)
    model.compile(optimizer=optimizers.Adam(1e-4), loss="binary_crossentropy", metrics=["accuracy"])

    model.fit(
        X_train, y_train, batch_size=32, epochs=epochs, validation_data=(X_val, y_val), class_weight=weights,
        callbacks=[callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)],
        verbose=2,
    )

    probs = model.predict(X_val, verbose=0)[:, 0]
    preds = (probs >= 0.5).astype(int)
    print(f"held-out images: {len(y_val)}, accuracy {np.mean(preds == y_val):.2f}")
    for name, label in classes.items():
        mask = y_val == label
        print(f"  {name}: {int((preds[mask] == label).sum())}/{int(mask.sum())} right")
    print("only a handful of held-out images, so treat this number as rough", flush=True)

    DATA_MODEL_PATH.parent.mkdir(exist_ok=True)
    model.save(DATA_MODEL_PATH)
    print(f"saved model to {DATA_MODEL_PATH}", flush=True)
    return model


if __name__ == "__main__":
    import sys
    # python CNN.py          trains on CIFAKE
    # python CNN.py --data   fine-tunes that model on the images in Data/
    train_on_data() if "--data" in sys.argv else train()
