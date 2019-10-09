import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import init
import numpy as np
import torchvision.models as models
from models_sceneparsing import ModelBuilder



def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        if m.weight.requires_grad:
            m.weight.data.normal_(std=0.02)
        if m.bias is not None and m.bias.requires_grad:
            m.bias.data.fill_(0)
    elif isinstance(m, nn.BatchNorm2d) and m.affine:
        if m.weight.requires_grad:
            m.weight.data.normal_(1, 0.02)
        if m.bias.requires_grad:
            m.bias.data.fill_(0)


def create_model(opt):
    ngpu = len(opt.gpu_ids)
    isEnhancer = opt.isEnhancer
    #isTest = opt.isTest

    class SGNResidualBlock(nn.Module):
        def __init__(self):
            super(SGNResidualBlock, self).__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(512 + 40, 512, 3, padding=1, bias=False),
                nn.InstanceNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, 3, padding=1, bias=False),
                nn.InstanceNorm2d(512)
            )

        def forward(self, seg_feat, att):
            att = att.unsqueeze(-1).unsqueeze(-1)
            att = att.repeat(1, 1, seg_feat.size(2), seg_feat.size(3))
            fusion = torch.cat((seg_feat, att), dim=1)
            return F.relu(seg_feat + self.encoder(fusion))

    class SGNResidualBlockHD(nn.Module):
        def __init__(self, n_feature):
            super(SGNResidualBlockHD, self).__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(n_feature + 40, n_feature, 3, padding=1, bias=False),
                nn.InstanceNorm2d(n_feature),
                nn.ReLU(inplace=True),
                nn.Conv2d(n_feature, n_feature, 3, padding=1, bias=False),
                nn.InstanceNorm2d(n_feature)
            )

        def forward(self, seg_feat, att):
            att = att.unsqueeze(-1).unsqueeze(-1)
            att = att.repeat(1, 1, seg_feat.size(2), seg_feat.size(3))
            fusion = torch.cat((seg_feat, att), dim=1)
            return F.relu(seg_feat + self.encoder(fusion))



 
    class SGNGenerator(nn.Module):
        def __init__(self, is_test=False):
            super(SGNGenerator, self).__init__()
            self.ngpu = ngpu


            self.encoder = nn.Sequential(
                nn.ReflectionPad2d(3),
                nn.Conv2d(108, 64, 7, padding=0),
                nn.InstanceNorm2d(64),
                nn.ReLU(inplace=True),

                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(128),
                nn.ReLU(inplace=True),

                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(256),
                nn.ReLU(inplace=True),

                nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(512),
                nn.ReLU(inplace=True),

            )

            self.res_block1 = SGNResidualBlock()
            self.res_block2 = SGNResidualBlock()
            self.res_block3 = SGNResidualBlock()
            self.res_block4 = SGNResidualBlock()
            self.res_block5 = SGNResidualBlock()

            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(256),
                nn.ReLU(inplace=True),

                nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(128),
                nn.ReLU(inplace=True),

                nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(64),
                nn.ReLU(inplace=True),

                nn.ReflectionPad2d(3),
                nn.Conv2d(64, 3, kernel_size=7, padding=0),
                nn.Tanh()

            )


            self.apply(init_weights)

        def forward(self, z, seg, att):

            seg = seg.type(torch.cuda.FloatTensor)
            z_seg = torch.cat((z, seg), 1)
            if isinstance(z_seg.data, torch.cuda.FloatTensor) and self.ngpu > 1:
                seg_feat = nn.parallel.data_parallel(self.encoder, z_seg, range(self.ngpu))
            else:
                seg_feat = self.encoder(z_seg)


            out1 = self.res_block1(seg_feat, att)
            out2 = self.res_block2(out1, att)
            out3 = self.res_block3(out2, att)
            out4 = self.res_block4(out3, att)
            out5 = self.res_block5(out4, att)

            if isinstance(out5.data, torch.cuda.FloatTensor) and self.ngpu > 1:
                output = nn.parallel.data_parallel(self.decoder, out5, range(self.ngpu))
            else:
                output = self.decoder(out5)

            return output


    class SGNGeneratorHD(nn.Module):

        def __init__(self, is_test=False):
            super(SGNGeneratorHD, self).__init__()
            self.ngpu = ngpu
            
            self.global_encoder = SGNGenerator().encoder
            self.global_res_block1 = SGNGenerator().res_block1
            self.global_res_block2 = SGNGenerator().res_block2
            self.global_res_block3 = SGNGenerator().res_block3
            self.global_res_block4 = SGNGenerator().res_block4
            self.global_res_block5 = SGNGenerator().res_block5
            global_decoder = SGNGenerator().decoder
            global_decoder = [global_decoder[i] for i in
                              range(len(global_decoder) - 3)]  # remove final global generator layers
            self.global_decoder = nn.Sequential(*global_decoder)

            self.encoder = nn.Sequential(
                nn.ReflectionPad2d(3),
                nn.Conv2d(108, 32, kernel_size=7, padding=0),
                nn.InstanceNorm2d(32),
                nn.ReLU(inplace=True),

                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(64),
                nn.ReLU(inplace=True)
            )

            self.res_block1 = SGNResidualBlockHD(64)
            self.res_block2 = SGNResidualBlockHD(64)

            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(32),
                nn.ReLU(inplace=True),

                nn.ReflectionPad2d(3),
                nn.Conv2d(32, 3, kernel_size=7, padding=0),
                nn.Tanh()
            )

            self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
            self.downsample2 = nn.functional.interpolate
            self.apply(init_weights)

        def forward(self, z, seg, att):
            # encoder
            seg = seg.type(torch.cuda.FloatTensor)
            z_seg = torch.cat((z, seg), 1)

            if isinstance(z_seg.data, torch.cuda.FloatTensor) and self.ngpu > 1:
                seg_feat = nn.parallel.data_parallel(self.encoder, z_seg, range(self.ngpu))
            else:
                seg_feat = self.encoder(z_seg)


            z_downsampled = self.downsample2(z, scale_factor=0.5, mode='nearest')
            seg_downsampled = self.downsample2(seg, scale_factor=0.5, mode='nearest')
            z_seg_downsampled = torch.cat((z_downsampled, seg_downsampled), 1)

            global_img_feat = self.global_encoder(z_seg_downsampled)

            global_out1 = self.global_res_block1(global_img_feat, att)

            global_out2 = self.global_res_block2(global_out1, att)

            global_out3 = self.global_res_block3(global_out2, att)

            global_out4 = self.global_res_block4(global_out3, att)

            global_out5 = self.global_res_block5(global_out4, att)

            output_prev = self.global_decoder(global_out5)

               
            out1 = self.res_block1(seg_feat + output_prev, att)
            out2 = self.res_block2(out1, att)

            if isinstance(out2.data, torch.cuda.FloatTensor) and self.ngpu > 1:
                output = nn.parallel.data_parallel(self.decoder, out2, range(self.ngpu))
            else:
                output = self.decoder(out2)

            return output

    class MultiscaleDiscriminator(nn.Module):
        def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d,
                     use_sigmoid=False, num_D=3, getIntermFeat=False):
            super(MultiscaleDiscriminator, self).__init__()
            self.num_D = num_D
            self.n_layers = n_layers
            self.getIntermFeat = getIntermFeat

            for i in range(num_D):
                netD = SGNNLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
                if getIntermFeat:
                    for j in range(n_layers + 2):
                        setattr(self, 'scale' + str(i) + '_layer' + str(j), getattr(netD, 'model' + str(j)))
                else:
                    setattr(self, 'layer' + str(i), netD)

            self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
            self.downsampleSeg = nn.UpsamplingNearest2d(scale_factor=0.5)

        def singleD_forward(self, model, input, segmentation, attribute):
            if self.getIntermFeat:
                result = [input]
                for i in range(len(model)):
                    result.append(model[i](result[-1], attribute))
                return result[1:]
            else:
                return [model(input, segmentation, attribute)]

        def forward(self, input, segmentation, attribute):
            num_D = self.num_D
            result = []
            input_downsampled = input
            segmentation_downsampled = segmentation
            for i in range(num_D):
                if self.getIntermFeat:
                    model = [getattr(self, 'scale' + str(num_D - 1 - i) + '_layer' + str(j)) for j in
                             range(self.n_layers + 2)]
                else:
                    model = getattr(self, 'layer' + str(num_D - 1 - i))
                result.append(self.singleD_forward(model, input_downsampled, segmentation_downsampled, attribute))
                if i != (num_D - 1):
                    input_downsampled = self.downsample(input_downsampled)
                    segmentation_downsampled = self.downsampleSeg(segmentation_downsampled)
            return result



    class DiscriminatorFeature(nn.Module):
        def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d, use_sigmoid=False,
                     getIntermFeat=False):
            super(DiscriminatorFeature, self).__init__()
            self.getIntermFeat = getIntermFeat
            self.n_layers = n_layers
            self.ngpu = ngpu

            kw = 4
            padw = int(np.ceil((kw - 1.0) / 2))
            sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

            nf = ndf
            for n in range(1, n_layers):
                nf_prev = nf
                nf = min(nf * 2, 512)
                sequence += [[
                    nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                    norm_layer(nf), nn.LeakyReLU(0.2, True)
                ]]

            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
                norm_layer(nf),
                nn.LeakyReLU(0.2, True)
            ]]

            if use_sigmoid:
                sequence += [[nn.Sigmoid()]]

            if getIntermFeat:
                for n in range(len(sequence)):
                    setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))
            else:
                sequence_stream = []
                for n in range(len(sequence)):
                    sequence_stream += sequence[n]
                self.model = nn.Sequential(*sequence_stream)

            self.apply(init_weights)

        def forward(self,segmentation, attribute):
            if self.getIntermFeat:
                res = [input]
                for n in range(self.n_layers + 2):
                    model = getattr(self, 'model' + str(n))
                    res.append(model(res[-1]))
                return res[1:]
            else:
                input = torch.cat((segmentation, attribute),1)
          
                if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
                    img_feat = nn.parallel.data_parallel(self.model, input, range(self.ngpu))
                else:
                    img_feat = self.model(input)

                return img_feat

    class SGNNLayerDiscriminator(nn.Module):
        def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d, use_sigmoid=False,
                     getIntermFeat=False):
            super(SGNNLayerDiscriminator, self).__init__()
            self.ConditionFeature = DiscriminatorFeature(input_nc=48, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d, use_sigmoid=False,
                     getIntermFeat=False)
            self.getIntermFeat = getIntermFeat
            self.n_layers = n_layers
            self.ngpu = ngpu

            kw = 4
            padw = int(np.ceil((kw - 1.0) / 2))
            sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

            nf = ndf
            for n in range(1, n_layers):
                nf_prev = nf
                nf = min(nf * 2, 512)
                sequence += [[
                    nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                    norm_layer(nf), nn.LeakyReLU(0.2, True)
                ]]

            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
                norm_layer(nf),
                nn.LeakyReLU(0.2, True)
            ]]

            if use_sigmoid:
                sequence += [[nn.Sigmoid()]]

            if getIntermFeat:
                for n in range(len(sequence)):
                    setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))
            else:
                sequence_stream = []
                for n in range(len(sequence)):
                    sequence_stream += sequence[n]
                self.model = nn.Sequential(*sequence_stream)

            self.classifier = nn.Sequential(
                nn.Conv2d(512 * 2, 512, 1, stride=(1, 1), padding=0),
                nn.InstanceNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(512, 1, kernel_size=kw, stride=1, padding=padw)
            )

            self.apply(init_weights)

        def forward(self, input, segmentation, attribute):
            if self.getIntermFeat:
                res = [input]
                for n in range(self.n_layers + 2):
                    model = getattr(self, 'model' + str(n))
                    res.append(model(res[-1]))
                return res[1:]
            else:
                attribute = attribute.unsqueeze(-1).unsqueeze(-1)
                attribute = attribute.repeat(1, 1, input.size(2), input.size(3))
 
                if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
                    img_feat = nn.parallel.data_parallel(self.model, input, range(self.ngpu))
                else:
                    img_feat = self.model(input)

                cond_feat = self.ConditionFeature(segmentation, attribute)
                fusion = torch.cat((img_feat, cond_feat), dim=1)
                if isinstance(fusion.data, torch.cuda.FloatTensor) and self.ngpu > 1:
                    classout = nn.parallel.data_parallel(self.classifier, fusion, range(self.ngpu))
                else:
                    classout = self.classifier(fusion)

                return classout


    if isEnhancer:
        G = SGNGeneratorHD()
    else:
        G = SGNGenerator()

    D = MultiscaleDiscriminator(input_nc=3)
    return G, D


from torchvision import models

ngpu = 2


class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.ngpu = ngpu
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        # h_relu1 = self.slice1(X)
        if isinstance(X.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            h_relu1 = nn.parallel.data_parallel(self.slice1, X, range(self.ngpu))
        else:
            h_relu1 = self.slice1(X)

        # h_relu2 = self.slice2(h_relu1)
        if isinstance(h_relu1.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            h_relu2 = nn.parallel.data_parallel(self.slice2, h_relu1, range(self.ngpu))
        else:
            h_relu2 = self.slice2(h_relu1)

        # h_relu3 = self.slice3(h_relu2)
        if isinstance(h_relu2.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            h_relu3 = nn.parallel.data_parallel(self.slice3, h_relu2, range(self.ngpu))
        else:
            h_relu3 = self.slice3(h_relu2)

        # h_relu4 = self.slice4(h_relu3)
        if isinstance(h_relu3.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            h_relu4 = nn.parallel.data_parallel(self.slice4, h_relu3, range(self.ngpu))
        else:
            h_relu4 = self.slice4(h_relu3)

        # h_relu5 = self.slice5(h_relu4)
        if isinstance(h_relu4.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            h_relu5 = nn.parallel.data_parallel(self.slice5, h_relu4, range(self.ngpu))
        else:
            h_relu5 = self.slice3(h_relu4)

        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        # out = [h_relu3, h_relu4, h_relu5]

        return out


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.cuda.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)


class VGGLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


class PerceptualLoss(nn.Module):
    def __init__(self, gpu_ids, args):
        super(PerceptualLoss, self).__init__()
        self.criterion = nn.L1Loss()
        builder = ModelBuilder()
        net_encoder = builder.build_encoder(arch=args.arch_encoder,fc_dim=args.fc_dim,weights=args.weights_encoder).cuda()
        self.net_encoder = net_encoder.eval()

        for p in self.net_encoder.parameters():
            p.requires_grad = False

       
    def forward(self, real, fake):
        xc = Variable(real.data.clone(), volatile=True)
        m = nn.AvgPool2d(3, stride=2, padding=1)
        f_fake = self.net_encoder(m(fake))
        f_real = self.net_encoder(m(xc))
        f_xc_c = Variable(f_real.data, requires_grad=False)
        loss = self.criterion(f_fake, f_xc_c)  

        return loss
