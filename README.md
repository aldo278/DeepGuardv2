## DeepGuard v2

This is the second iteration of the DeepGuard project, which aims to detect AI-generated images using various machine learning techniques.
Unfortunately, video data and metadata is around 16 GB of data, and since my laptop sucks and I don't think lab pc's will be able to handle that amount of data, this project will only focus on images.

I will be using the following dataset from kaggle:
- CIFAKE: https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images

The last version used an API that i tweaked so that it took 5 frames from a video, where it eventually correctly predicted the fake video , so I might be able to implement something similar.

I will be giving updates on my progress in this file, and will be adding new sections as I go along. Newest updates will be at the top.

---------------------------------------------------------------------------------


## Detector
Before using a model, I decided to first check if The given image contains a hidden watermark, as this would be a good indicator of whether the image is real or fake.

watermark.py was originally named data.py but i decided to use the same file to detect a watermark

The training data images are too small thus the watermark detector library cannot make out the image, let alone check for a hidden watermark. Skip this for now, I will try to add a loop where i give a UI fake images and annotate them myself. Moving on to Logistic regression...
