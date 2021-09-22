import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import nnio
import cv2
from torchvision import models

# Overwrite these values before launching script
SAVE_PATH = 'dream.avi'

FPS = 24

# Accelerate calculations
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Number of iterations & learning rate
NUM_EPOCHS = 500
LR = 0.03

# Result picture dimensions
H = 720
W = 1280

CAM_MATRIX = np.array([
    [1000, 0.0, W / 2],
    [0.0, 1000, H / 2],
    [0.0, 0.0,    1.0]
])
DISTORTION = np.array([0.001, 0.02, 0.0, 0.0])

# Load and preprocess input picture
img = np.random.random(size=[H, W, 3])

# Make a copy of input picture
orig_img = img

# Define forward hooks
hook_out = {}
def hook(module, input, output):
    hook_out['feats'] = output

# Create model
model = models.vgg19(pretrained=True)
model.requires_grad_ = False
model.to(DEVICE)
model.eval()

# Register forward hooks
layer_id = 34
#model.features[layer_id].register_forward_hook(hook)
#for i in range(layer_id + 1, len(model.features)):
#    model.features[i] = nn.Identity()
#model.classifier = nn.Identity()

model.classifier.register_forward_hook(hook)

running_average = None

out_video = cv2.VideoWriter(SAVE_PATH, cv2.VideoWriter_fourcc('M','J','P','G'), FPS, (W, H))

epoch = 0
while True:
    img_small = cv2.resize(img, (500, 350))
    img_small = torch.tensor(img_small.transpose(2, 0, 1)[None], dtype=torch.float32, device=DEVICE, requires_grad=True)

    # Make forward pass
    out = model(img_small)

    # Get features
    feats = hook_out['feats']

    # Compute loss
    neuron_id = (epoch // (FPS * 10)) % feats.shape[1]
    loss = torch.mean(feats[0, neuron_id]) # .clamp(max=10)

    # Compute gradients
    loss.backward()
    grad = img_small.grad
    img = torch.tensor(img.transpose(2, 0, 1)[None], dtype=torch.float32)
    grad = grad.detach().cpu()

    # Normalize gradients
    grad_avg = torch.sqrt((grad**2).mean()).cpu().numpy()
    if running_average is None:
        running_average = grad_avg
    running_average = running_average * 0.99 + grad_avg * 0.01
    grad = grad / running_average
    grad = F.interpolate(grad, size=(H, W), mode='bicubic', align_corners=True)
    grad = grad.clip(-1, 1)

    # Gradient ascent step
    img = img + grad * LR
    img = img.clip(0.0, 1.0)

    # To numpy
    img = img[0].numpy().transpose(1, 2, 0)
    
    # Distortion
    CAM_MATRIX[0,2] = W / 2 + np.random.normal() * 10
    CAM_MATRIX[1,2] = H / 2 + np.random.normal() * 10
    newcameramtx = CAM_MATRIX.copy()
    newcameramtx[:2,:2] *= 1.002
    img = cv2.undistort(img, CAM_MATRIX, DISTORTION, None, newcameramtx)
    
    # Rotate colors
    hsv = cv2.cvtColor(img[:,:,::-1], cv2.COLOR_BGR2HSV)
    hsv[:, :, 0] = (hsv[:, :, 0] + 2) % 360
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[:,:,::-1].copy()

    print('step:', epoch, 'seconds:', int(epoch / FPS), 'neuron', neuron_id, ' loss:', loss.detach().cpu().numpy())
    
    cv2.imshow('dream', img[:,:,::-1])
    out_video.write((img[:,:,::-1] * 255).astype('uint8'))
    if cv2.waitKey(1) == ord('q'):
        break

    epoch += 1


# Closes all the frames
out_video.release()
cv2.destroyAllWindows()
