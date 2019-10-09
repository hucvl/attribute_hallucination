import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import Variable
import torchvision.transforms as transforms
from torchvision.utils import save_image


from model import create_model
from model import GANLoss
from model import VGGLoss, PerceptualLoss
from data import SGNDatasetTest
import random
import PIL
import os
import scipy.io as sio
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--img_root', type=str, required=True,
                    help='root directory that contains images')
parser.add_argument('--model_path', type=str, required=True,
                    help='checkpoint file')
parser.add_argument('--save_dir', type=str, required=True,
                    help='checkpoint file')
parser.add_argument('--num_threads', type=int, default=4,
                    help='number of threads for fetching data (default: 4)')
parser.add_argument('--batch_size', type=int, default=1,
                    help='batch size (default: 1)')
parser.add_argument('--no_cuda', action='store_true',
                    help='do not use cuda')
parser.add_argument('--isEnhancer', action='store_true',
                    help='Enhancer Model')

parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

parser.add_argument('--manualSeed', type=int, help='manual seed')

args = parser.parse_args()

if not args.no_cuda and not torch.cuda.is_available():
    print("WARNING: You have not a CUDA device")
    args.no_cuda = True

if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
print("Random Seed: ", args.manualSeed)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)

gpu_ids = []
for str_id in args.gpu_ids.split(','):
    id = int(str_id)
    if id >= 0:
        gpu_ids.append(id)
args.gpu_ids = gpu_ids
if len(args.gpu_ids) > 0:
    torch.cuda.set_device(args.gpu_ids[0])
    torch.cuda.manual_seed_all(args.manualSeed)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def init_z_foreach_layout(category_map, batchsize):
    numofseg = 150
    print(category_map.size())
    ZT = torch.FloatTensor(batchsize, 100, category_map.size(1), category_map.size(2))
    ZT.fill_(0.0)
    ZT = ZT.cuda()
    z = torch.FloatTensor(batchsize, 100, 1, 1).cuda()
    for j in range(numofseg + 1):

        mask = category_map.eq(j)

        if (mask.any()):

            z = torch.rand(batchsize, 100, 1, 1).cuda()
            z.resize_(batchsize, 100, 1, 1).normal_(0, 1)
            z = z.expand(batchsize, 100, category_map.size(1), category_map.size(2))
            mask = mask.unsqueeze(1)
            mask = mask.type(torch.FloatTensor)
            ZT = ZT.add_(z * mask.cuda())

    noise = torch.rand(batchsize, 100, category_map.size(1), category_map.size(2)).cuda()
    return ZT + noise

def colorencode(category_im):
    category_im = category_im[0]
    colorcodes = sio.loadmat(args.img_root + "/color150.mat")
    colorcodes = colorcodes['colors']
    idx = np.unique(category_im)
    h, w = category_im.shape
    colorCodeIm = np.zeros((h, w, 3)).astype(np.uint8)
    for i in range(idx.shape[0]):
        if idx[i] == 0:
            continue
        b = np.where(category_im == idx[i])
        rgb = colorcodes[idx[i] - 1]
        bgr = rgb[::-1]
        colorCodeIm[b] = rgb
    return colorCodeIm


if __name__ == '__main__':

    print('Loading a dataset...')
    test_data = SGNDatasetTest(args)
    test_loader = data.DataLoader(test_data,
                                   batch_size=args.batch_size,
                                   shuffle=False,
                                   num_workers=args.num_threads)

    # pretrained model
    print('Loading a pretrained model...')
    G, _ = create_model(args)
  

    
    ind = args.model_path.rfind("/")
    model_name = args.model_path[ind + 1 :]
    G.load_state_dict(torch.load(args.model_path))
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    if not os.path.exists(args.save_dir + '/real'):
        os.mkdir(args.save_dir + '/real')
    if not os.path.exists(args.save_dir + '/' + model_name):
        os.mkdir(args.save_dir + '/' +  model_name)
    if not os.path.exists(args.save_dir + '/colorseg'):
        os.mkdir(args.save_dir + '/' +  '/colorseg')
    if not os.path.exists(args.save_dir + '/annotation'):
        os.mkdir(args.save_dir + '/' +  '/annotation')

    gt_attributes = np.zeros((1338, 40))

    if not args.no_cuda:
        G.cuda()

    for i, (img, att, seg, cat) in enumerate(test_loader):
            bs = img.size(0)
            gt_attributes[i,:] = att.data.numpy()
            seg = seg.type(torch.FloatTensor)
            img = Variable(img.cuda())
            att = Variable(att.cuda())
            seg = Variable(seg.cuda())
            cat = Variable(cat.cuda())
            if i == 0:
                seg1 = seg
                att1 = att
                cat1 = cat
                img1 = img
            cat_np = cat.data.cpu().numpy()
            colorseg = colorencode(cat_np)
            Z = init_z_foreach_layout(cat, bs)
            img_norm = img * 2 - 1
            img_G = img_norm
            fake = G(Z, seg, att)
            save_image((fake.data + 1) * 0.5, args.save_dir + '/' + model_name +'/%d.png' % (i + 1))
            save_image((img_G.data + 1) * 0.5, args.save_dir + '/real/%d.png' % (i + 1))
            colorim = Image.fromarray(colorseg)
            annot = Image.fromarray(cat_np[0])
            colorim.save(args.save_dir + '/colorseg/%d.png' % (i + 1))
            annot.save(args.save_dir + '/annotation/%d.png' % (i + 1))
    np.save(args.save_dir + '/' + 'test_attributes', gt_attributes)
