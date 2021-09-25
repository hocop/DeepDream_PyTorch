import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision import models
from PIL import Image, ImageFont, ImageDraw 
import sys

# Overwrite these values before launching script
SAVE_PATH = sys.argv[1] if len(sys.argv) > 1 else 'dream.avi'

FPS = 24
CHANGING_PERIOD = 20

# Read labels
exec(f'labels = {open("labels.txt").read()}')
FONT_SIZE = 30
font = ImageFont.truetype("arial.ttf", FONT_SIZE)


# Accelerate calculations
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Learning rate
LR = 0.04

# Result picture dimensions
H = 1080
W = 1920

CAM_MATRIX = np.array([
    [1000, 0.0, W / 2],
    [0.0, 1000, H / 2],
    [0.0, 0.0,    1.0]
])
DISTORTION = np.array([0.001, 0.01, 0.0, 0.0])

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

def convert_relu_to_leaky(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, nn.LeakyReLU(0.05))
            print('relu')
        else:
            convert_relu_to_leaky(child)

convert_relu_to_leaky(model)

out_video = cv2.VideoWriter(SAVE_PATH, cv2.VideoWriter_fourcc(*'MP4V'), FPS, (W, H))

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

running_average = None
grad_accum = None
epoch = 0

neuron_1 = neuron_2 = neuron_3 = neuron_4 = None

while True:
    img_small = cv2.resize(img, (768, 432))
    img_small = torch.tensor(img_small.transpose(2, 0, 1)[None], dtype=torch.float32, device=DEVICE, requires_grad=True)

    # Make forward pass
    delta = int(0.05 * img_small.shape[3])
    lower_w = img_small.shape[3] // 2 - delta
    upper_w = img_small.shape[3] // 2 + delta
    lower_h = img_small.shape[2] // 2 - delta
    upper_h = img_small.shape[2] // 2 + delta
    img_two = torch.cat([
        img_small[:, :, :upper_h, :upper_w], # top left
        img_small[:, :, :upper_h, lower_w:], # top right
        img_small[:, :, lower_h:, :upper_w], # bottom left
        img_small[:, :, lower_h:, lower_w:], # bottom right
    ], 0)
    out = model(img_two)

    # Compute loss
    mod = FPS * CHANGING_PERIOD
    if epoch % mod == 0:
        neuron_1 = np.random.choice(list(labels.keys()))
    if (epoch - mod // 4) % mod == 0 or neuron_2 is None:
        neuron_2 = np.random.choice(list(labels.keys()))
    if (epoch - 2 * mod // 4) % mod == 0 or neuron_3 is None:
        neuron_3 = np.random.choice(list(labels.keys()))
    if (epoch - 3 * mod // 4) % mod == 0 or neuron_4 is None:
        neuron_4 = np.random.choice(list(labels.keys()))
    loss = torch.mean(out[0, neuron_1]) + torch.mean(out[1, neuron_2]) + torch.mean(out[2, neuron_3]) + torch.mean(out[3, neuron_4])

    # Compute gradients
    loss.backward()
    grad = img_small.grad.detach().cpu()
    img = torch.tensor(img.transpose(2, 0, 1)[None], dtype=torch.float32)

    # Normalize gradients
    grad_norm = torch.sqrt((grad**2).mean(1, keepdim=True)).cpu().numpy() + 1e-6
    # grad_norm = torch.abs(grad).mean().cpu().numpy()
    if running_average is None:
        running_average = grad_norm
    running_average = running_average * 0.99 + grad_norm * 0.01
    grad = grad / running_average
    grad = grad.clip(-1, 1)

    # To numpy
    img = img[0].numpy().transpose(1, 2, 0)
    grad = grad[0].numpy().transpose(1, 2, 0)

    # Distortion
    CAM_MATRIX[0,2] = CAM_MATRIX[0,2] * 0.99 + np.random.normal(W / 2, 100) * 0.01
    CAM_MATRIX[1,2] = CAM_MATRIX[1,2] * 0.99 + np.random.normal(H / 2, 100) * 0.01
    newcameramtx = CAM_MATRIX.copy()
    newcameramtx[:2,:2] *= 1.002
    img = cv2.undistort(img, CAM_MATRIX, DISTORTION, None, newcameramtx)
    img = rotate_image(rotate_image(img, 0.02), -0.02)

    # Accumulate gradients
    grad = cv2.resize(grad, (W, H), interpolation=cv2.INTER_LANCZOS4)
    if grad_accum is None:
        grad_accum = grad
    grad_accum = 0.05 * grad + 0.95 * grad_accum
    grad_accum = cv2.undistort(grad_accum, CAM_MATRIX, DISTORTION, None, newcameramtx)

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
    # Upper
    text_1 = f'{labels[neuron_1].split(",")[0]}'.upper()
    text_2 = f'{labels[neuron_2].split(",")[0]}'.upper()
    w, h = image_editable.textsize(text_2, font=font)
    text_x = max(W - w, 0)
    text_y = 1
    image_editable.text((1, text_y), text_1, (237, 230, 211), font=font)
    image_editable.text((text_x, text_y), text_2, (237, 230, 211), font=font)
    # Lower
    text_1 = f'{labels[neuron_3].split(",")[0]}'.upper()
    text_2 = f'{labels[neuron_4].split(",")[0]}'.upper()
    w, h = image_editable.textsize(text_2, font=font)
    text_x = max(W - w, 0)
    text_y = H - FONT_SIZE - 1
    image_editable.text((1, text_y), text_1, (237, 230, 211), font=font)
    image_editable.text((text_x, text_y), text_2, (237, 230, 211), font=font)
    # Make text faded
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
