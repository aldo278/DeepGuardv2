## DeepGuard v2

This is the second iteration of the DeepGuard project, which aims to detect AI-generated images using various machine learning techniques.
Unfortunately, video data and metadata is around 16 GB of data, and since my laptop sucks and I don't think lab pc's will be able to handle that amount of data, this project will only focus on images.

I will be using the following dataset from kaggle:
- CIFAKE: https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images

The last version used an API that i tweaked so that it took 5 frames from a video, where it eventually correctly predicted the fake video , so I might be able to implement something similar.

I will be giving updates on my progress in this file, and will be adding new sections as I go along. Newest updates will be at the top.

---------------------------------------------------------------------------------


## Sandbox and Backend
Built a local sandbox for testing the detection models. `Detector/` now acts as a backend (with FastAPI) and the sandbox page calls it over HTTP. The browser extension will call the same API later.

Run it from `Detector/`: `..\.venv\Scripts\uvicorn server:app --host 127.0.0.1 --port 8000`, then open http://127.0.0.1:8000 (API docs at `/docs`).
New dependencies: `pip install fastapi uvicorn python-multipart httpx`.
### What was built
- `watermark.py` and `logisticRegression.py` no longer download, train or run demos when imported. Training is in `train()` and demos are under `__main__`.
- `logisticRegression.py` saves the trained model to `Detector/models/` (gitignored) and has `predict(bytes)`. Any image size is shrunk to 32x32 first, since the model was trained on CIFAKE.
- `watermark.py` has `predict(bytes)`, which can only say FAKE (Stable Diffusion watermark found) or UNKNOWN. A missing watermark does not make an image real. The original `check_sd_watermark` decodes 32 bits, which can never contain "StableDiffusion", so the new version decodes 136 bits.
- `registry.py` is the one place models are listed. Every model takes image bytes and returns `{model, verdict, prob_fake, detail, ms}`, so adding the CNN later is one line.
- `server.py` routes: `GET /health`, `GET /models`, `POST /predict` (image upload, optional `models` list), `POST /evaluate` (scores everything in `Data/Real` and `Data/Fake`), and `/` serves the Sandbox.
- `Sandbox/` (plain HTML/JS): drop, browse or paste images, see each model's verdict, probability bar and timing. Mark an image Real or Fake to get a check or cross per model and a running accuracy. The "Evaluate Data/" button shows a confusion matrix per model. UNKNOWN and ERROR results are counted as abstained, not as wrong.

### Results so far
`Data/` has 1 real and 21 fake images.

| Model | Result | Time per image |
|---|---|---|
| watermark | UNKNOWN on every image (no Stable Diffusion watermark found) | 154 ms |
| logreg | 20 of 22 correct (91%), but it called the one real image FAKE | 22 ms |

The 91% only reflects the 21-to-1 imbalance. Logreg mostly says FAKE to everything and is often fully confident (probability 1.0), so these numbers say little until there are more real images.

### Known issues
- `watermark_precheck` calls `check_c2pa`, which does not exist yet.
- `python logisticRegression.py` took about 11 minutes even with cached features. Its closing demo crashes because `Detector/TestImages/` is gone (the model is saved before that point).
- I have not opened the Sandbox in a browser yet. Only the server routes and the JavaScript syntax were checked.

### Next
1. Gather more real images. Nothing can be judged until `Data/Real` has more than one.
2. Train the CNN, save it, and add it to the registry (CIFAKE 32x32 or my own `Data/` images is still undecided).
3. Add an `image_url` route for the extension, with limits (http/https only, size cap, timeout, no private IP ranges), and restrict CORS to the extension's origin.
4. Handle full-size images properly. Squashing them to 32x32 is probably why logreg does so badly, so this likely means retraining on my own images or a model that takes different sizes.
5. Build the extension itself. Video is still out of scope.


## Detector
Before using a model, I decided to first check if The given image contains a hidden watermark, as this would be a good indicator of whether the image is real or fake.

watermark.py was originally named data.py but i decided to use the same file to detect a watermark

The training data images are too small thus the watermark detector library cannot make out the image, let alone check for a hidden watermark. Skip this for now, I will try to add a loop where i give a UI fake images and annotate them myself. Moving on to Logistic regression...


### Logistic Regression
The current model started with a 68% accuracy, its okay for being the first take but not ready as MVP. 
Another thing to note is that the script logisticRegression.py takes around 5 min to run and return results, it might be because of the large size of the dataset.
Sigmoid function: activation function that squashes the output between 0 and 1, making it suitable for binary classification.

More changes to this model. 

FFT: Fast Fourier Transformer: an efficient algorithm for computien the Fourier Transform, which takes a signal and tells you which frequencies (wave patterns) its made of and how strong each one is. For example, when someone plays a piano chord, the sound is a combined wave, your ear can still pickup on individual notes. The Forier Transform does that mathematically: give it a combined wave and it separates out pure sine waves that add up to it, along with how loud each one is.
Image Application.
An image is 2D signal, so you use the 2D version (fft2). Here, 'frequency' means how quickly brightness changes as you move across the image:
- Low frequencies: gradual changes, like a smooth sky, blurry gradient, or the overall shape and lighting.
- High frequencies: rapid changes, like sharp edges, fine texture, grain, and noise.





### Convolutional Neural Network (CNN)
I will try to implement a CNN model with the current dataset, and see if it can achieve a higher accuracy.
If it fails, I will try to get my own data. Either way making my own dataset with current images would help not only on accuracy but also on the watermark detection.


I did some research and found that CNNs are implement with either TensorFlow or PyTorch. I will try to use TensorFlow first, as it is more beginner-friendly.

Definitions:
Tensor: a tensor is a container for numbers arranged along some number of dimensions. It is the basic data structure in TensorFlow, and its essentially the same idea as a numpy array.
with extra features for running on GPU's and tracking gradients. So a 3d matrix would be a 3d tensor.

Convolution: a mathematical operation that slides a small grid of numbers, called a kernel (or filter), over an input array (such as an image or time-series data) to 
extract local patterns and spatial relationships.
What the convolution does, step by step:
- take a small filter (a grid of weights, here 3x3)
- place it over a 3x3 patch of the image
- multiply each weight by the pixel under it, then add everything up, that gives one number
- slide the filter over and repeat, across the whole image.
The numbers collected from all positions form one feature map: a new image showing where that filter pattern was found


Convolutional Base: The first part of a CNN, which is a stack of Conv2D and MaxPooling2D layers that extract features from the image. In this project it can find differences between neighboring pixels. (face isolation is a separate step). Some features include edges, gradients, color contrasts, and simple textures, middle layers can detect more complex patterns like corners, repeated patterns, skin, or fur or fabric textures. Deeper layers get larger, task specific features like facial features, objects, or text.

Rectified Linear Unit (ReLU): an activation function in neural networks that outputs the input directly if it is positve, and outputs zero if the input is negative or zero
In a Con2D layer, this is applied to every value in every feature map, right after the filter's weighted sum. It is an activation function that introduces non-linearity into the model, allowing it to learn more complex patterns. Kind of like the sigmoid function.

For the CNN, there are algorithms that help in adjust weights to minimize error.
- Adam optimizer: a popular optimization algorithm that adapts the learning rate for each parameter based on estimates of the first and second moments of the gradients
- sgd (stochastic gradient descent): uses one fixed step size for everything, whereas Adam's default learning rate is much smaller (0.001)


will it differeciate between CGI and AI?
check if too smooth or too irregular for CGI
sobel operators
https://en.wikipedia.org/wiki/Sobel_operator

What is an Epoch????
An Epoch is one full pass through the entire training set. With 50,000 training images, one epoch means the model has seen all 50,000 once (in batches of 32, so about 1563 weight updates). In the first NN, we set epochs=10m so it made 10 passes.

Why more than one pass? A single look at the data isn't enough for the weights to settle on good values. Each pass improves them a bit more, like re-reading a textbook chapter several times. Too many epochs, though, ans the model starts memorizing the training images instead of learning general patterns (overfitting), which is exactly what the plot helps spot.

![logo](https://365datascience.com/resources/blog/thumb@1024_bpm7kc63z8-overfitting-vs-underfitting-models-examples.webp)

### Claude
I will be using Claude Code to go over my implementations, it will mainly be used as another engineer to double check my work.


### Annotator
I planned on using a GUI annotator so I can drop in images and label them as either fake or real. It did not work, so I deleted the folder and moved the Data subfolder into the main folder. I have also gathered some fake images that vary in size, so I need to make the model work with the different sized images. I still need to gather some real sample images.

### Extension and Testing
Before improving the models, I want to have an interface to test the model. The final product is still a browser extension, like the first version, where you simply click on a video or image to detect if the input is fake or not. First, I am going to make a sandbox for testing.


### Results
overfitted, model started memorizing the data, not learning.
I need more data, or i might use a pretrained model from hugging face

The CNN now contains 10 layers 

Resume this session with:
claude --resume da2b3c65-3449-4841-8f22-7fbb9f236377