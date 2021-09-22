import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision import models
from PIL import Image, ImageFont, ImageDraw 

# Overwrite these values before launching script
SAVE_PATH = 'dream.avi'

FPS = 24

# Read labels
FONT_SIZE = 20
exec(f'labels = {open("labels.txt").read()}')
font = ImageFont.truetype("arial.ttf", FONT_SIZE)


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

out_video = cv2.VideoWriter(SAVE_PATH, cv2.VideoWriter_fourcc('M','J','P','G'), FPS, (W, H))

running_average = None
grad_accum = None
epoch = 0

while True:
    img_small = cv2.resize(img, (500, 350))
    img_small = torch.tensor(img_small.transpose(2, 0, 1)[None], dtype=torch.float32, device=DEVICE, requires_grad=True)

    # Make forward pass
    out = model(img_small)

    # Get features
    feats = hook_out['feats']

    # Compute loss
#    neuron_id = (epoch // (FPS * 15)) % feats.shape[1]
    if epoch % (FPS * 15) == 0:
        neuron_1 = np.random.randint(feats.shape[1])
        neuron_2 = np.random.randint(feats.shape[1])
    loss = torch.mean(feats[0, neuron_1]) + torch.mean(feats[0, neuron_2])

    # Compute gradients
    loss.backward()
    grad = img_small.grad.detach().cpu()
    img = torch.tensor(img.transpose(2, 0, 1)[None], dtype=torch.float32)

    # Normalize gradients
    grad_avg = torch.sqrt((grad**2).mean()).cpu().numpy()
    if running_average is None:
        running_average = grad_avg
    running_average = running_average * 0.99 + grad_avg * 0.01
    grad = grad / running_average
    grad = F.interpolate(grad, size=(H, W), mode='bicubic', align_corners=True)
    grad = grad.clip(-1, 1)

    # To numpy
    img = img[0].numpy().transpose(1, 2, 0)
    grad = grad[0].numpy().transpose(1, 2, 0)

    # Distortion
    CAM_MATRIX[0,2] = W / 2 + np.random.normal() * 10
    CAM_MATRIX[1,2] = H / 2 + np.random.normal() * 10
    newcameramtx = CAM_MATRIX.copy()
    newcameramtx[:2,:2] *= 1.002
    img = cv2.undistort(img, CAM_MATRIX, DISTORTION, None, newcameramtx)
    
    # Accumulate gradients
    if grad_accum is None:
        grad_accum = grad
    grad_accum = 0.1 * grad + 0.9 * cv2.undistort(grad_accum, CAM_MATRIX, DISTORTION, None, newcameramtx)

    # Gradient ascent step
    img = img + grad_accum * LR
    img = img.clip(0.0, 1.0)

    # Rotate colors
    hsv = cv2.cvtColor(img[:,:,::-1], cv2.COLOR_BGR2HSV)
    hsv[:, :, 0] = (hsv[:, :, 0] + 2) % 360
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[:,:,::-1].copy()

    print('step:', epoch, 'seconds:', int(epoch / FPS), 'neuron', neuron_1, neuron_2, ' loss:', loss.detach().cpu().numpy())

    # Draw text
    img_show = Image.fromarray((img * 255).astype('uint8'))
    image_editable = ImageDraw.Draw(img_show)
    text_1 = f'{labels[neuron_1].split(",")[0]}    '.upper()
    text_2 = f'&    {labels[neuron_2].split(",")[0]}'.upper()
    text = text_1 + text_2
    w, h = image_editable.textsize(text_1, font=font)
    w = w + image_editable.textsize('&', font=font)[0] / 2
    text_x = max(W / 2 - w, 0)
    text_y = H - FONT_SIZE - 1
    image_editable.text((text_x, text_y), text, (237, 230, 211), font=font)
    img_show = np.array(img_show) / 255
    img_show = img_show * 0.8 + img * 0.2

    cv2.imshow('dream', img_show[:,:,::-1])
    out_video.write((img_show[:,:,::-1] * 255).astype('uint8'))
    if cv2.waitKey(1) == ord('q'):
        break

    epoch += 1


# Closes all the frames
out_video.release()
cv2.destroyAllWindows()
