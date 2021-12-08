import os, time, pickle, argparse, networks, utils
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import transforms
import torch.nn.functional as F
from tqdm import tqdm
import cv2
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--in_ngc', type=int, default=3, help='input channel for generator')
parser.add_argument('--out_ngc', type=int, default=3, help='input channel for generator')
parser.add_argument('--ngf', type=int, default=32)
parser.add_argument('--input_size', type=int, default=256, help='input size')
parser.add_argument('--pre_trained_model', required=False, default="Result/HWGAN_IAM_dataset/weights/generator_latest.pkl", help='pre_trained cartoongan model path')
parser.add_argument('--image_dir', required=False, default='data/test_language', help='test image path')
parser.add_argument('--output_image_dir', required=False, default='result_image', help='output test image path')
parser.add_argument('--is_cuda', type=str, default='True', help='indicate CUDA usage [True, False]')
args = parser.parse_args()

# Device
if args.is_cuda == 'False':
    device = 'cpu'
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = True

print("DEVICES = {}".format(device))


# Load generator
G = networks.generator(args.in_ngc, args.out_ngc, args.ngf)
if torch.cuda.is_available():
    G.load_state_dict(torch.load(args.pre_trained_model))
else:
    G.load_state_dict(torch.load(args.pre_trained_model, map_location=lambda storage, loc: storage))
G.to(device)

src_transform = transforms.Compose([
        #transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

# Create output folder
if not os.path.exists(args.output_image_dir):
    os.mkdir(args.output_image_dir)


# Inference
name_list = os.listdir(args.image_dir)
name_list = [f for f in name_list if ('.jpg' in f) or ('.png' in f)]

for name in tqdm(name_list):
    load_path = os.path.join(args.image_dir, name)
    #print(load_path)
    save_out_path = os.path.join(args.output_image_dir, name)

    raw_image = cv2.imread(load_path)[:,:,::-1]
    #print(raw_image.shape)
    #w = raw_image.shape[1]
    #raw_image = raw_image[:,int(w/2):,:]
    
    raw_image = Image.fromarray(raw_image)
    x = src_transform(raw_image).to(device)
    #x = F.interpolate(x, 64, 1024)
    x = torch.unsqueeze(x, 0)
    G_recon = G(x)[0]
    plt.imsave(save_out_path[:-3] + 'png', (G_recon.cpu().detach().numpy().transpose(1, 2, 0) + 1) / 2)

# from FID.fid_score import calculate_fid_given_paths

# fid_value = calculate_fid_given_paths(path_A='FID/test_data/CartoonGAN', path_B='FID/test_data/testA',
#                                       batch_size=64,
#                                       cuda=True,
#                                       dims=2048)
# print('FID: ', fid_value)