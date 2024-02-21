import time
from orion.perception.extractor.lseg import load_lseg_for_inference
import torch
import cv2
import numpy as np

# Load pre-trained model
checkpoint_path = "/data/daiyp/cv_models/lseg_demo_e200.ckpt"
device = torch.device("cuda:0")
model = load_lseg_for_inference(checkpoint_path, device)

# Encode pixels to CLIP features
# rgb of shape (batch_size, H, W, 3) in RGB order
# pixel_features of shape (batch_size, 512, H, W)
rgb = cv2.imread("data/experiments/fbe_4ok3usBNeis/recordings/rgb/4ok3usBNeis_22.png")
rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
# rgb = cv2.resize(rgb, (480, 640))
rgb = np.expand_dims(rgb, axis=0)

tstart = time.time()
for _ in range(50):
    pixel_features = model.encode(rgb)
tend = time.time()
print("time: ", (tend - tstart) / 50)

# Decode pixel CLIP features to text labels - we can introduce new labels
# at inference time
# one_hot_predictions of shape (batch_size, H, W, len(labels))
# visualizations of shape (batch_size, H, W, 3)
labels = ["other", "floor", "wall", "bed", "lamp", "cabinet", "audio speaker"]
one_hot_predictions, visualizations = model.decode(pixel_features, labels)

# Visualize the predictions

import matplotlib.pyplot as plt

f, axarr = plt.subplots(4, 2, figsize=(9, 14))
print(rgb[0].shape)
axarr[0, 0].imshow(rgb[0].astype(int))
print(visualizations[0].shape)
axarr[0, 1].imshow(visualizations[0])
plt.savefig("lseg.png")
