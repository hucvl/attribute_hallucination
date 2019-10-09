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
from data import SGNDataset
import random
import PIL
import os

parser = argparse.ArgumentParser()
parser.add_argument('--img_root', type=str, required=True,
                    help='root directory that contains images')
parser.add_argument('--save_filename', type=str, required=True,
                    help='checkpoint file')
parser.add_argument('--num_threads', type=int, default=4,
                    help='number of threads for fetching data (default: 4)')
parser.add_argument('--num_epochs', type=int, default=100,
                    help='number of threads for fetching data (default: 600)')
parser.add_argument('--batch_size', type=int, default=32,
                    help='batch size (default: 64)')
parser.add_argument('--learning_rate', type=float, default=0.0002,
                    help='learning rate (dafault: 0.0002)')
parser.add_argument('--lr_decay', type=float, default=0.5,
                    help='learning rate decay (dafault: 0.5)')
parser.add_argument('--momentum', type=float, default=0.5,
                    help='beta1 for Adam optimizer (dafault: 0.5)')
parser.add_argument('--isEnhancer', action='store_true',
                    help='use enhancer Generator')
parser.add_argument('--resume_train', action='store_true',
                    help='continue training from the latest epoch')
parser.add_argument('--isTest', action='store_true',
                    help='test')

parser.add_argument('--gpu_ids', type=str, default='0',
                    help='gpu ids: e.g. 0  0,1,2')

parser.add_argument('--manualSeed', type=int,
                    help='manual seed')
parser.add_argument('--coarse_model', required=True, 
                    help='folder to model path')
# Scene Parsing Model related arguments

parser.add_argument('--scene_parsing_model_path', required=True,
                    help='folder to model path')
parser.add_argument('--suffix', default='_best.pth',
                    help="which snapshot to load")
parser.add_argument('--arch_encoder', default='resnet34_dilated8',
                    help="architecture of net_encoder")
parser.add_argument('--fc_dim', default=2048, type=int,
                    help='number of features between encoder and decoder')

args = parser.parse_args()
args.weights_encoder = os.path.join(args.scene_parsing_model_path, 'encoder' + args.suffix)

if not torch.cuda.is_available():
    print("WARNING: You have not a CUDA device")

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

    ZT = torch.FloatTensor(batchsize, 100, 512, 512)
    ZT.fill_(0.0)
    ZT = ZT.cuda()

    for j in range(numofseg + 1):

        mask = category_map.eq(j)

        if (mask.any()):
            z = torch.rand(batchsize, 100, 1, 1).cuda()
            z.resize_(batchsize, 100, 1, 1).normal_(0, 1)
            z = z.expand(batchsize, 100, 512, 512)
            mask = mask.unsqueeze(1)
            mask = mask.type(torch.FloatTensor)
            ZT = ZT.add_(z * mask.cuda())

    del mask, z, category_map
    return ZT


if __name__ == '__main__':
    print('Loading a pretrained fastText model...')

    print('Loading a dataset...')
    train_data = SGNDataset(args)
    train_loader = data.DataLoader(train_data,
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   num_workers=args.num_threads)

    print('Loading SGN model...')
    G, D = create_model(args)

    pretrained_dict = torch.load(args.coarse_model)
    model_dict = G.state_dict()

    print("Pretrained Global Generator is loading...\n")
    for k, v in pretrained_dict.items():
        k_model = 'global_' + k
        if k_model in model_dict and v.size() == model_dict[k_model].size():
            print(k_model + "\n")
            model_dict[k_model] = v

    G.load_state_dict(model_dict)

    criterionGAN = GANLoss(use_lsgan=True)
    criterionFeat = torch.nn.L1Loss()
    #criterionVGG = VGGLoss(args.gpu_ids)
    criterionPercept = PerceptualLoss(args.gpu_ids, args)

    G.cuda()
    D.cuda()

    n_epoch_fixGlobal = 10
    d_optimizer = torch.optim.Adam(D.parameters(), lr=args.learning_rate, betas=(args.momentum, 0.999))

    if n_epoch_fixGlobal > 0:
        import sys

        if sys.version_info >= (3, 0):
            finetune_list = set()
        else:
            from sets import Set

            finetune_list = Set()

        params_dict = dict(G.named_parameters())
        params = []
        for key, value in params_dict.items():
            if not key.startswith('global'):
                params += [value]
                finetune_list.add(key.split('.')[0])
        print(
            '------------- Only training the local enhancer network (for %d epochs) ------------' % n_epoch_fixGlobal)
        print('The layers that are finetuned are ', sorted(finetune_list))
        g_optimizer = torch.optim.Adam(params, lr=args.learning_rate, betas=(args.momentum, 0.999))

    start_epoch = 0
    if args.resume_train:
        rf = open("logHD.txt",'r')
        log = rf.readline()
        log = log.split(' ')
        start_epoch = int(log[0])
        print('Resuming pretrained models...')
        pretrained_dict = torch.load(args.save_filename + "_G_latest")
        model_dict = G.state_dict()
        for k, v in pretrained_dict.items():
            if k in model_dict and v.size() == model_dict[k].size():
                model_dict[k] = v
            else:
                print(k + "\n") 
        G.load_state_dict(model_dict)
        D.load_state_dict(torch.load(args.save_filename + "_D_latest"))

    if start_epoch >= n_epoch_fixGlobal:
        g_optimizer = torch.optim.Adam(G.parameters(), lr=args.learning_rate, betas=(args.momentum, 0.999))


    for epoch in range(start_epoch, args.num_epochs):

        # training loop
        avg_D_real_loss = 0
        avg_D_real_m_loss = 0
        avg_D_real_m2_loss = 0
        avg_D_fake_loss = 0
        avg_G_fake_loss = 0
        avg_percept_loss = 0
        #avg_vgg_loss = 0
        avg_percept_loss = 0
        for i, (img, att, seg, cat, nnseg) in enumerate(train_loader):

            bs = img.size(0)

            rnd_batch_num = np.random.randint(len(train_data), size=bs)
            rnd_att_list = [train_data[i][1] for i in rnd_batch_num]
            rnd_att_np = np.asarray(rnd_att_list)
            rnd_att = torch.from_numpy(rnd_att_np).float()

            seg = seg.type(torch.FloatTensor)
            nnseg = nnseg.type(torch.FloatTensor)
            img = Variable(img.cuda())
            att = Variable(att.cuda())
            rnd_att = Variable(rnd_att.cuda())
            seg = Variable(seg.cuda())
            nnseg = Variable(nnseg.cuda())
            cat = Variable(cat.cuda())
            Z = init_z_foreach_layout(cat, bs)
            img_norm = img * 2 - 1
            img_G = img_norm

            # UPDATE DISCRIMINATOR
            requires_grad(G, False)
            requires_grad(D, True)
            D.zero_grad()

            # real image with relevant layout and attribute
            real_logit = D(img_norm, seg, att)
            real_loss = criterionGAN(real_logit, True)
            avg_D_real_loss += real_loss.data.item()
            real_loss.backward()

            # real image with mismatching layout
            real_m_logit = D(img_norm, nnseg, att)
            real_m_loss = 0.25 * criterionGAN(real_m_logit, False)
            avg_D_real_m_loss += real_m_loss.data.item()
            real_m_loss.backward()

            # real image with mismatching attribute
            real_m2_logit = D(img_norm, seg, rnd_att)
            real_m2_loss = 0.25 * criterionGAN(real_m2_logit, False)
            avg_D_real_m2_loss += real_m2_loss.data.item()
            real_m2_loss.backward()


            # synthesized image with relevant layout and attribute
            fake = G(Z, seg, att)
            fake_logit = D(fake.detach(), seg, att)
            fake_loss = 0.5 * criterionGAN(fake_logit, False)
            avg_D_fake_loss += fake_loss.data.item()
            fake_loss.backward()

            d_optimizer.step()

            # UPDATE GENERATOR
            requires_grad(G, True)
            requires_grad(D, False)
            G.zero_grad()
            fake = G(Z, seg, att)
            fake_logit = D(fake, seg, att)
            fake_loss = criterionGAN(fake_logit, True)
            #vgg_loss =10 * criterionVGG(img_G, fake)
            percept_loss =10 * criterionPercept(img_G, fake)
            avg_G_fake_loss += fake_loss.data.item()
            #avg_vgg_loss += vgg_loss.data.item()
            avg_percept_loss += percept_loss.data.item()
            G_loss = fake_loss + percept_loss
            G_loss.backward()
            g_optimizer.step()

            if i % 10 == 0:
                print('Epoch [%d/%d], Iter [%d/%d], D_real: %.4f, D_misSeg: %.4f, D_misAtt: %.4f, D_fake: %.4f, G_fake: %.4f, Percept: %.4f'
                      % (epoch + 1, args.num_epochs, i + 1, len(train_loader), avg_D_real_loss / (i + 1),
                         avg_D_real_m_loss / (i + 1), avg_D_real_m2_loss / (i + 1), avg_D_fake_loss / (i + 1), avg_G_fake_loss / (i + 1),
                         avg_percept_loss / (i + 1)))
                save_image((fake.data + 1) * 0.5, './examples/%d_fake_hd.png' % (epoch + 1))
                save_image((img_G.data + 1) * 0.5, './examples/%d_real_hd.png'% (epoch + 1))
                torch.save(G.state_dict(), args.save_filename + "_G_latest")
                torch.save(D.state_dict(), args.save_filename + "_D_latest")
                log_file=open("logHD.txt","w")
                log_file.write(str(epoch)+" "+str(i))
                log_file.close()

        if epoch == n_epoch_fixGlobal:
            print("Training coarse and enhancer networks together...")
            g_optimizer = torch.optim.Adam(G.parameters(), lr=args.learning_rate, betas=(args.momentum, 0.999))

        if epoch % 1 == 0:
            torch.save(G.state_dict(), args.save_filename + "_G_" + str(epoch))
            torch.save(D.state_dict(), args.save_filename + "_D_" + str(epoch))
