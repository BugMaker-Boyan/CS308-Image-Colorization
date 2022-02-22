import torch

from model.DCGAN import Generator

import cv2
import numpy as np

device = torch.device('cuda:0')

generator = torch.nn.DataParallel(Generator()).cuda()

generator.load_state_dict(torch.load("./gen_model_14"))
image = cv2.imread("tiny-imagenet-200/train/n01443537/images/n01443537_2.JPEG")
cv2.imwrite("./ori.JPEG", image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_input = torch.tensor(np.reshape(gray, (1, 1, 64, 64)) / 128.0 - 1, dtype=torch.float)
out = generator(gray_input)
out = ((out + 1) * 128).cpu().detach().numpy().astype('uint8')
out = np.reshape(out, (3, 64, 64))
L, a, b = out
image = cv2.merge((L, a, b))
bgr_out = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
cv2.imwrite("./pic.JPEG", bgr_out)
cv2.imwrite("./gray.JPEG", L)
