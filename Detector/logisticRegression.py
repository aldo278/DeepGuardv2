#lets us download the dataset from kagglehub
import kagglehub

# a computer vision library. I am going to use it to read image files into arrays, it can also read color channels as BGR
import cv2

# for arrays and math, numpy arrays are the same as tensors
import numpy as np

# path represents file paths as objects intead of plain strings
from pathlib import Path

# runs a function over many items using multiple threads at once. In this case loading and processing thousands of image files in parallel
from concurrent.futures import ThreadPoolExecutor

# the SGDC classifier is a linear classifier trained with stochastic gradient descent (SGD) which updates the weights using small batches or single samples, (like Adam)
from sklearn.linear_model import SGDClassifier

# chains several steps into one object. used to combine the scaler and classifier.
from sklearn.pipeline import make_pipeline

# rescales each feature to have a mean 0 and standard deviation 1. This is the same motivation as dividing by 255. but stronger, since it makes every feature comparable in scale
from sklearn.preprocessing import StandardScaler

# evaluation tools, accuracy_score for fraction of predictions that were correct, classification_report for precision/recall/f1, confusion_matrix for a table of true vs predicted labels
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# here we download the dataset from kagglehub or we use the cached copy 
path = Path(kagglehub.dataset_download("birdy654/cifake-real-and-ai-generated-synthetic-images"))
CACHE = Path(__file__).parent / "cache"
FEATURES = "pixels_fft" # part of the cache filename, change it whenever read_features changes

# an image is just a matrix of pixel values.
# 32x32 RGB -> flatten into a 3072-length vector -> one "row" for the model.
# a linear model can't find texture artifacts in raw pixels, so we also add the
# log-magnitude FFT of the grayscale image (1024 values): generated images tend
# to have unusual high-frequency patterns that show up in the spectrum.
def read_features(fname):
    #reads the image int a np array of shape (32, 32, 3) with integer values 0-255. we convert fname into a string which openCV wants, channels come back in BGR order 
    img = cv2.imread(str(fname))
    
    if img is None:
        return None

    # collapses 3 color channels into 1 brightness channel, giving shape (32, 32). The FFT is run on a single 2D grid, so it needs one channel
    # .astype(np.float32) converts the pixel values from integers 0-255 to floats, which is what the FFT function expects
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # line that does most of the work, easiest read inside out. 
    # np.fft.fft2(gray) the 2D fft. it re-describes the image as a sum of waves instead of pixels, each output value says "how much of this particular wave pattern is in the image"
    # low frequencies are smooth, gradual changes in brightness. High frequencies are fine in detail, sharp edges, and noise. The output is a grid of the same shape, but values are complex numbers
    # np. fft.fftshift() reorders the grid so the lowest frequencies sits in the center and higher frequencies radiate outward. its only a rearrangement, but it makes the spectrum consistent and easy to interpret
    # np.abs() takes the absolute value of each complex num
    # np.log1p() computes log(1+x). The magnitudes span a huge range (the lowest frequency value) is enourmous compared to the rest), so the log compresses them into a manageable scale
    # result is a 32x32 grid:1024 numbers describing which frequencies are present and how strongly
    spectrum = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(gray))))

    # img.flatten() unrolls the colo image into a vector of 32x32x3 = 3072 values (flatten from CNN)
    # spectrum.flatten() does the same for the spectrum giving 1024 values
    # np.hstack() joins them side by side into one vector of len 4096

    return np.hstack([img.flatten(), spectrum.flatten()]).astype(np.float32)


def load_split(split_dir):
    # these two empty lists will hold the feature vectors, and y will hold their labels
    X, y = [], []

    # we loop over two (number, name) pairs, unpacking each into label and folder. This defines the labeling, which fits well with the sigmoid activation function.
    for label, folder in [(1, "FAKE"), (0, "REAL")]:

        # split_dir / folder builds a path like ../train/FAKE ..iterdir() yields every file in the folder and list() collects them all
        files = list((split_dir / folder).iterdir())

        # creates a pool of worker threads, and the with block shuts it down cleanly when finished.
        with ThreadPoolExecutor() as pool: # imread is I/O + JPEG decode, threads help a lot

            # pool.map() runs and read_features() on every file, spreading the work across the threads, and hands back the results in the original order. Threads help here because 
            # reading and decoding image files spends a lot of time waiting on the disk, and openCV releases Python's lock while it does it. 
            # Img in this loop is really the 4096 len feature vector (or none), not an image. The name is a leftover
            for img in pool.map(read_features, files):

                # skips any file that failed to load, and otherwise stores the vector along with its label
                if img is not None:
                    X.append(img)
                    y.append(label)
    # converts the lists to numpy arrays, where X becomes a 2D matric of shape (num_images, 4096) (one row per image, one column per feature) and y becomes a 1D array of 0, and 1s                
    return np.array(X), np.array(y)

# name is the parameter for the split: either "train" or "test". 
def get_split(name):
    # this builds two file paths and unpacks them into two variables. CACHE is the cache folder, FEATURES is the feature type, and name is the split name. The result is a path like cache/X_train_pixels_fft.npy
    cache_x, cache_y = CACHE / f"X_{name}_{FEATURES}.npy", CACHE / f"y_{name}_{FEATURES}.npy"

    # if the feature file is already on disk, load both arrays and return immediately, np.load() reads a npy dile, Numpy's native binary format, straight back into an array 
    # with the same shape and dtype. Much faster than re-reading 100,000 images and recomputring the fft's. The return ends the function
    if cache_x.exists():
        return np.load(cache_x), np.load(cache_y)

    #the cache-miss path. path/ name builds the dataset folder (like .../train) using the path variable from the top, and load_split() does the full pipeline
    X, y = load_split(path / name)

    # creates the cache folder if it does not exist. 
    CACHE.mkdir(exist_ok=True)

    # writes both arrays to disk as .npy files for the next time
    np.save(cache_x, X)
    np.save(cache_y, y)

    # hands back the freshly computed arrays, so the function returns the same thing whetjer it hit the cache or not
    return X, y


# calls the caching function from before. the first run takes a while (it extracts features and saves them), and later runs load from the disk. The result is a feature matrix 
# and a lable array for each split
X_train, y_train = get_split("train")
X_test, y_test = get_split("test")

print(f"train: {X_train.shape}, test: {X_test.shape}", flush=True)

# SGDClassifier with log_loss is logistic regression trained via SGD.
# lbfgs would need many full passes over 100k images; SGD converges in a handful of epochs. lbfgs is also scikit-learn's Logistic regression
# SGD is sensitive to feature scale, and pixels (0-255) and FFT magnitudes live on very
# different scales, so StandardScaler gives every feature mean 0 / std 1 first.
# inside the pipeline the scaler only learns its mean/std from the training data.
model = make_pipeline(
    StandardScaler(),              # replaced from 1000   
    SGDClassifier(loss="log_loss", max_iter=500, tol=1e-3, alpha=1e-5, average=True, random_state=42),
)

# training runs in two stages through the pipeline: standard scaler computes the mean and sd of each of the 4096 columns on the training data, and transforms the training data with them
model.fit(X_train, y_train)

# scakes the test data using the training set's mean and sd, then the classifier makes a call per image. Internally it computes the probability of fake and thresholds it at .5
# giving it at .5 or 1
preds = model.predict(X_test)
# print(f"accuracy: {accuracy_score(y_test, preds):.4f}")
# print(confusion_matrix(y_test, preds))
# print(classification_report(y_test, preds, target_names=["REAL", "FAKE"]))


# function from claude. to test if it returns a result
def predict_one(path):
    img = cv2.imread(str(path))
    img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    spectrum = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(gray))))
    x = np.hstack([img.flatten(), spectrum.flatten()]).astype(np.float32)
    return model.predict_proba(x.reshape(1, -1))[0, 1]  # probability of fake

# real image
print(f'Real Image, probability that is fake: {predict_one("Z:\\Capstone\\DeepGuardv2\\Detector\\TestImages\\ac2b4c7e-2f69-4b49-96a1-28be8ea39a4e.jpg")}')

# fake image of damian :)
print(f'Probability of Damian fake image: {predict_one("Z:\\Capstone\\DeepGuardv2\\Detector\\TestImages\\cd9f9a60-d135-4a96-9d9f-4c59d4dde0f9.png")}')
