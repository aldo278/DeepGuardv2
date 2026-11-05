# the one place detectors are listed. Each entry is a function: image bytes in, result dict out (see PLAN.md section 4).
# Imports are lazy so the server starts fast and one broken model (for example a missing saved file) doesn't take the others down.
def _watermark(image_bytes):
    import watermark
    return watermark.predict(image_bytes)


def _logreg(image_bytes):
    import logisticRegression
    return logisticRegression.predict(image_bytes)


def _cnn(image_bytes):
    import CNN
    return CNN.predict(image_bytes)


MODELS = {
    "watermark": _watermark,
    "logreg": _logreg,
    "cnn": _cnn,
}


# runs the chosen models on one image. A model that fails reports an ERROR result for itself instead of failing the whole request
def run(image_bytes, names=None):
    results = []
    for name in names or MODELS:
        try:
            results.append(MODELS[name](image_bytes))
        except ValueError:
            raise  # unreadable image, the same for every model
        except Exception as e:
            results.append({"model": name, "verdict": "ERROR", "prob_fake": None, "detail": f"{type(e).__name__}: {e}", "ms": 0})
    return results
