
vernum = __import__('sys').version_info[:1]

if(vernum[0]==3):
	from tkinter import *
	from tkinter import font
	from tkinter import filedialog
	from tkinter import ttk
elif(vernum[0]==2):
	from Tkinter import *
	import tkFont as font
	import tkFileDialog as filedialog
	import ttk
	import matlab.engine


import scipy.io as sio
import numpy as np
from PIL import Image, ImageDraw, ImageTk
import io
import sys
sys.path.append('.')
print(sys.path)
import os
import argparse
import json
from time import time
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.nn import init
import functools
from torch.autograd import Variable
import torchvision.utils as vutils
import torchvision.transforms as transforms
from scipy.stats import truncnorm
import PIL


from model import create_model
import gc
import time


from fastphoto import process_stylization
from fastphoto.photo_wct import PhotoWCT


from WCT2.transfer import WCT2
from WCT2.utils.io import Timer, open_image, load_segment, compute_label_info
from torchvision.utils import save_image

#Semantic Segmentation Module
from semantic_segmentation_pytorch.models import ModelBuilder, SegmentationModule
from semantic_segmentation_pytorch.lib.nn import user_scattered_collate, async_copy_to
from semantic_segmentation_pytorch.dataset import TestDataset
import semantic_segmentation_pytorch.lib.utils.data as torchdata
from semantic_segmentation_pytorch.lib.utils import as_numpy, mark_volatile


parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--imageSize', type=int, default=128, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--model_path', default='./pretrained_models/sgn_rnm_percept', help="pretrained model path")
parser.add_argument('--attribute_path', default='./fixed_attribute.npy', help="pretrained model path")
parser.add_argument('--isEnhancer', action='store_true', help='use enhancer Generator')
parser.add_argument('--image_size', type=int, default=256)
parser.add_argument('--isTest', action='store_true', help='test')

#Semantic Segmentation Module Options
parser.add_argument('--imgSize', default=[300, 400, 500, 600],
                        nargs='+', type=int,
                        help='list of input image sizes.'
                             'for multiscale testing, e.g. 300 400 500')
parser.add_argument('--imgMaxSize', default=1000, type=int,
                        help='maximum input image size of long edge')
parser.add_argument('--padding_constant', default=8, type=int,
                        help='maxmimum downsampling rate of the network')
parser.add_argument('--segm_downsampling_rate', default=8, type=int,
                        help='downsampling rate of the segmentation label')


opt = parser.parse_args()
print(opt)

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

gpu_ids = []
for str_id in opt.gpu_ids.split(','):
    id = int(str_id)
    if id >= 0:
        gpu_ids.append(id)
opt.gpu_ids = gpu_ids
if len(opt.gpu_ids) > 0:
    torch.cuda.set_device(opt.gpu_ids[0])
    torch.cuda.manual_seed_all(opt.manualSeed)


cudnn.benchmark = True


ngpu = len(opt.gpu_ids)
nz = opt.nz
nc = 3

netG, netD = create_model(opt)
model_path = opt.model_path
input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
attribute = torch.FloatTensor(opt.batchSize, 40, 1, 1)
segment = torch.FloatTensor(opt.batchSize, 8, opt.imageSize, opt.imageSize)
category = torch.FloatTensor(opt.batchSize, 1, opt.imageSize, opt.imageSize)
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
netGDict = torch.load(model_path)

model_dict = netG.state_dict()
for k, v in netGDict.items():
        if k in model_dict and v.size() == model_dict[k].size():
            print(k + "\n")
            model_dict[k] = v
netG.load_state_dict(model_dict)
netG = netG.eval()

if ngpu>0:
    netG.cuda()
    attribute = attribute.cuda()
    segment = segment.cuda()
    noise = noise.cuda()
    category = category.cuda()
image_size = opt.image_size
imname = '1'

imoriginal_path = "./Test/test/"+imname+".jpg"
imcolor_path ="./Test/test"+imname+"/content_seg.png"
imgray_path = "./Test/test"+imname+"/content_grayseg.png"



attribute_name = '1234'
att_path = "./Attributes/"+attribute_name+'.npy'



class DrawingWindow():
	def __init__(self, imcolor_path, imoriginal_path, imgray_path, image_size):

		self.root = Tk()
		self.root.configure(background='gray')
		self.root.title("SGN")
		self.canvas = Canvas(self.root)
		self.scroll_x = Scrollbar(self.root, orient="horizontal", command=self.canvas.xview)
		self.frame = Frame(self.canvas)
		self.frame.configure(background='gray')
		self.image_size = image_size

		#Drawing Region
		self.lbl_darea = Label(self.frame, text="Layout", font=("Helvetica", 18), bg="gray")
		self.lbl_darea.grid(row = 0, column = 1)
		self.drawing_area = Canvas(self.frame, width=self.image_size, height=self.image_size, cursor="tcross")
		self.drawing_area.grid(row=1, column=1)
		self.drawing_area.bind("<Motion>", self.motion)
		self.drawing_area.bind("<ButtonPress-1>", self.b1down)
		self.drawing_area.bind("<ButtonRelease-1>", self.b1up)
		self.drawing_area.configure(background='white')

		self.lbl_garea = Label(self.frame, text="Hallucinated", font=("Helvetica", 18), bg="gray")
		self.lbl_garea.grid(row = 0, column = 3)
		self.lbl_empty = Label(self.frame, text="   ", font=("Helvetica", 18), bg="gray")
		self.lbl_empty.grid(row = 1, column = 0)
		self.generated_area = Canvas(self.frame, width=self.image_size, height=self.image_size)
		self.generated_area.grid(row=1, column=3)
		self.generated_area.configure(background='white')
		self.lbl_empty2 = Label(self.frame, text="  ", font=("Helvetica", 18), bg="gray")
		self.lbl_empty2.grid(row = 0, column = 6)


		self.transient_attribute = np.zeros((1, 40))
		colorsMat = sio.loadmat('color150.mat')
		self.colors = colorsMat['colors']
		att_file = open('attributes.txt', 'r')
		self.attributes = att_file.readlines()
		self.noise_count = 1
		binarycodesMat = sio.loadmat('binarycodes.mat')
		self.binarycodes = binarycodesMat['binarycodes']
		objectsMat = sio.loadmat('objectName150.mat')
		objects = objectsMat['objectNames']
		self.objects = objects[:, 0].tolist()


		self.b1 = "up"
		self.xold, self.yold = None, None
		self.coords = []
		self.first = 0
		self.drawn_object = []
		self.drawn_object_index = []
		self.selected_object = 0
		self.selected_attribute = -1

		style1 = ttk.Style()
		self.lb_attribute_frame=Frame(self.frame)
		self.lb_attribute_frame.grid(row=4, column=3)
		self.attribute_lb = Listbox(self.lb_attribute_frame, font=("Garamond", 15),relief="raised",borderwidth = '4')
		self.attribute_lb.pack(side = 'left',fill = 'y' )
		self.scrollbar_att = Scrollbar(self.lb_attribute_frame, orient=VERTICAL,command=self.attribute_lb.yview)
		self.scrollbar_att.pack(side="right", fill="y")
		self.attribute_lb.config(yscrollcommand=self.scrollbar_att.set)
		self.attribute_lb.bind('<<ListboxSelect>>', self.onselectAttribute)
		self.attribute_lb.bind('<Motion>', self.mouseonAttribute)
		self.attribute_lb.bind('<Leave>', self.on_leave)	

		self.lbl_style = Label(self.frame, text="Style Transfer Method", font=("Helvetica", 18), bg="gray")
		self.lbl_style.grid(row = 2, column = 5)
		options= ["WCT2","FPST"]
		self.stylemethods_cb = ttk.Combobox(self.frame, values=options, font=("Helvetica", 16))
		self.stylemethods_cb.grid(row=3, column=5)
		self.stylemethods_cb.current(1)
		self.stylemethods_cb.bind("<<ComboboxSelected>>", self.onselectStyleMethod)
		bigfont = font.Font(family="Helvetica",size=16)
		self.selected_stylemethod = 0
		self.stylemethods_cb.option_add("*TCombobox*Listbox*Font", bigfont)

		self.lb_objects_frame=Frame(self.frame)
		self.lb_objects_frame.grid(row=4, column=1)
		self.lb_objects = Listbox(self.lb_objects_frame,font=("Garamond", 15),relief="raised",borderwidth = '4')
		self.lb_objects.pack(side="left", fill="y")
		self.scrollbar_obj = Scrollbar(self.lb_objects_frame, orient=VERTICAL,command=self.lb_objects.yview)
		self.scrollbar_obj.pack(side="right", fill="y")
		self.lb_objects.config(yscrollcommand=self.scrollbar_obj.set)
		self.lb_objects.bind('<<ListboxSelect>>', self.onselect)
		self.lb_objects.bind('<Motion>', self.mouseonObjects)
		self.lb_objects.bind('<Leave>', self.on_leave_objects)

		self.lbcurrent =-1
		self.lbobjcurrent =-1

		for item in self.objects:
		    self.lb_objects.insert(END, str(item)[2:len(str(item)) - 1])
		for item in self.attributes:
		    self.attribute_lb.insert(END, str(item)[:len(str(item)) - 1])

		# Buttons
		ttk.Style().configure("TButton", padding=1, relief="raised", background="#aaa",borderwidth = '4', activebackground="#ddd",font=("Helvetica Neue", 14))
		self.generate_button = ttk.Button(self.frame, text='Generate', command=self.generateCallBack)
		self.generate_button.grid(row=2,column=1)
		self.Z_button = ttk.Button(self.frame, text='Random Attribute', command=self.DifferentAttributeCallBack)
		self.Z_button.grid(row=5, column=3)
		self.Att_button = ttk.Button(self.frame, text='Random Noise', command=self.DifferentZCallBack)
		self.Att_button.grid(row=6, column=3)
		self.increase_button = ttk.Button(self.frame, text='increase', command=self.increase_attribute_CallBack)
		self.increase_button.grid(row=2, column=3)
		self.decrease_button = ttk.Button(self.frame, text='decrease', command=self.decrease_attribute_CallBack)
		self.decrease_button.grid(row=3, column=3)
		self.transfer_button = ttk.Button(self.frame, text='Transfer', command=self.transferCallBack)
		self.transfer_button.grid(row=1, column=4)

		self.imtk = None
		self.iforiginal = False
		self.img = Image.open(imcolor_path)
		self.imgGray = Image.open(imgray_path)
		self.img_original = Image.open(imoriginal_path)
		self.img_original, self.img_original_color, self.img_original_gray = self.transformImage2(self.img_original, self.imgGray)
		self.img_original_resized, self.img_original_color_resized, self.img_original_gray_resized = self.transformImage(self.img_original, self.imgGray)
		
		self.img.save('./' + imname + '_LayColor.png', "PNG")
		self.imgGray.save('./' + imname + '_LayGray.png', "PNG")
		self.img_original.save('./' + imname + '_original.png', "PNG")
		self.img_original_color_resized.save('./' + imname + '_LayColorResized.png', "PNG")
		self.img_original_gray_resized.save('./' + imname + '_LayGrayResized.png', "PNG")
		self.img_original_resized.save('./' + imname + '_originalResized.png', "PNG")

		self.segmentBinary = self.binaryCodedImage(self.img_original_gray_resized)
		self.objectcategories = np.reshape(np.array(list(self.img_original_gray_resized.getdata())), (self.img_original_gray_resized.size[0], self.img_original_gray_resized.size[1]))
		cat_np = self.objectcategories
		cat = torch.from_numpy(cat_np).float()
		category.resize_as_(cat.cuda()).copy_(cat)
		self.noise = self.init_z(category, 1)
		self.imseg = ImageTk.PhotoImage(image=self.img_original_color_resized)
		self.imorg = ImageTk.PhotoImage(image=self.img_original)

		self.lbl_oarea = Label(self.frame, text="Input", font=("Helvetica", 18), bg="gray")
		self.lbl_oarea.grid(row = 0, column = 5)
		self.original_area = Canvas(self.frame, width=self.img_original.size[0], height=self.img_original.size[1])
		self.original_area.grid(row=1, column=5)
		self.original_area.configure(background='white')
		self.drawing_area.create_image(self.image_size / 2 + 1, self.image_size / 2 + 1, image=self.imseg)
		self.original_area.create_image(self.img_original.size[0] / 2 + 1, self.img_original.size[1] / 2 + 1, image=self.imorg)
		self.original_area.update_idletasks()



		self.menubar = Menu(self.frame,font=("Helvetica", 16))
		self.filemenu = Menu(self.menubar, tearoff=0)
		self.filemenu.add_command(label="New Image", font=("Helvetica", 14), command=self.openNewImage)
		self.menubar.add_cascade(label="File", menu=self.filemenu)
		self.root.config(menu=self.menubar)

		self.canvas.create_window(0, 0, anchor='nw', window=self.frame)
		self.canvas.update_idletasks()
		self.canvas.configure(scrollregion=self.canvas.bbox('all'), 
				 xscrollcommand=self.scroll_x.set)		 
		self.canvas.pack(fill='both', expand=True, side='top')
		self.scroll_x.pack( fill=BOTH, side='top')
		self.canvas.configure(scrollregion=self.canvas.bbox('all'), 
				 xscrollcommand=self.scroll_x.set)


		screen_width = self.root.winfo_screenwidth()
		screen_height = self.root.winfo_screenheight()

		self.load_segmentation_module()
		self.root.geometry(str(screen_width)+"x"+str(screen_height))
		self.root.mainloop()

	def increase_attribute_CallBack(self):
		#print("increase")
		self.transient_attribute[0, self.selected_attribute] = self.transient_attribute[0, self.selected_attribute] + 0.3
		self.generateCallBack()

	def DifferentZCallBack(self):
		self.noise = self.init_z(category, 1)
		#np.save('fixed_noise', self.noise.cpu().numpy())
		print("Noise initialized")
		self.generateCallBack()

	def DifferentAttributeCallBack(self):
		rndatt = np.random.randint(1338, size=1)
		#print(rndatt)
		rnd_attribute = np.load("./Attributes/"+str(rndatt[0])+'.npy')
		self.transient_attribute = np.expand_dims(rnd_attribute, axis=0)
		self.generateCallBack()

	def decrease_attribute_CallBack(self):
		#print("decrease")
		self.transient_attribute[0, self.selected_attribute] = self.transient_attribute[0, self.selected_attribute] - 0.3
		self.generateCallBack()

	def truncated_normal(self, size, threshold=1):
		values = truncnorm.rvs(-threshold, threshold, size=size)
		return values
	def transformImage2(self, image, seg):
		# Resize
		graycode = np.array(seg)
		colorcode = self._colorencode(graycode)
		graycode = Image.fromarray(graycode)
		colorcode = Image.fromarray(colorcode, 'RGB')
		return image, colorcode, graycode

	def transformImage(self, image, seg):
		# Resize
		resize = transforms.Resize(self.image_size)
		image = resize(image)
		resize = transforms.Resize(self.image_size, interpolation=PIL.Image.NEAREST)
		seg = resize(seg)
		# Center crop
		crop = transforms.CenterCrop((self.image_size, self.image_size))
		image = crop(image)
		seg = crop(seg)
		graycode = np.array(seg)
		colorcode = self._colorencode(graycode)
		graycode = Image.fromarray(graycode)
		colorcode = Image.fromarray(colorcode, 'RGB')
		return image, colorcode, graycode

	def load_segmentation_module(self):
		model_path = "./semantic_segmentation_pytorch/baseline-resnet50_dilated8-ppm_bilinear_deepsup"
		suffix = "_epoch_20.pth"
		arch_encoder = 'resnet50_dilated8'
		arch_decoder = 'ppm_bilinear_deepsup'
		fc_dim = 2048
		num_class = 150
		num_val = -1


		weights_encoder = os.path.join(model_path,
                                        'encoder' + suffix)
		weights_decoder = os.path.join(model_path,
                                        'decoder' + suffix)
		builder = ModelBuilder()
		net_encoder = builder.build_encoder(
		    arch=arch_encoder,
		    fc_dim=fc_dim,
		    weights=weights_encoder)
		net_decoder = builder.build_decoder(
		    arch=arch_decoder,
		    fc_dim=fc_dim,
		    num_class=num_class,
		    weights=weights_decoder,
		    use_softmax=True)

		crit = nn.NLLLoss(ignore_index=-1)

		self.segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)
		self.segmentation_module.cuda()
		self.segmentation_module.eval()
		print("Segmentation Module was created!")
		

	def _colorencode(self, category_im):
		colorcodes = sio.loadmat("./color150.mat")
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


	def init_z(self, categories, batchsize):
		noise.resize_(batchsize, 100, self.image_size, self.image_size).normal_(0, 1)
		return noise

	def generateCallBack(self):
		self.generated_area.delete("all")
		print(self.selected_attribute)
		if self.selected_attribute==-1:
			att_np = np.load(att_path)
			att_np = np.expand_dims(att_np, axis=0)
			self.transient_attribute = att_np
			np.save('fixed_attribute', self.transient_attribute)
			self.transient_attribute = np.load('./fixed_attribute.npy')

		np.save('fixed_attribute', self.transient_attribute)
		seg_np = self.segmentBinary
		seg_np = seg_np[np.newaxis]
		seg_np = np.transpose(seg_np, (0, 3, 1, 2))
		seg = torch.from_numpy(seg_np).float()
		segment.resize_as_(seg.cuda()).copy_(seg)

		att_np = self.transient_attribute[:, :]
		att = torch.from_numpy(att_np).float()
		attribute = att.cuda()


		cat_np = self.objectcategories
		cat = torch.from_numpy(cat_np).float()
		category.resize_as_(cat.cuda()).copy_(cat)

		fixed_noise = self.noise	
		sorted_inds = np.argsort(att_np[0])
		sorted_inds = sorted_inds[::-1]
		best_inds = sorted_inds[:5]
		for ix in best_inds:
			print(self.attributes[ix].strip() + ': ', self.transient_attribute[0,ix])

		
		if opt.isEnhancer:
			fake = netG(Variable(fixed_noise), Variable(segment), Variable(attribute))
		else:
			fake = netG(Variable(fixed_noise), Variable(segment), Variable(attribute))			
		imnp = fake.data.cpu().numpy()
		gen_im = self.inverse_transform(imnp)
		gen_im = gen_im[0]
		im = Image.fromarray(np.uint8(gen_im))
		im.save("./" + imname + '_G.png')
		self.noise_count = self.noise_count + 1
		self.imtk = ImageTk.PhotoImage(image=im)


		self.generated_area.create_image(self.image_size/2 + 1,self.image_size/2 + 1,image=self.imtk)
		self.generated_area.after(50)
		self.generated_area.update()
		gc.collect()
		print('Generation completed!')

	def transferCallBack(self):


		if(self.selected_stylemethod == 0):
			#WCT2
			transfer_at = set()

			transfer_at.add('encoder')

			transfer_at.add('decoder')

			transfer_at.add('skip')
			device = 'cuda:0'
			device = torch.device(device)
			wct2 = WCT2(transfer_at=transfer_at, option_unpool='cat5', device=device, verbose=False)
			content = open_image("./"+imname+"_original.png").to(device)
			style = open_image("./"+imname+"_G.png", self.image_size).to(device)
			content_segment = load_segment("./"+imname+"_LayGray.png", self.image_size)
			style_segment = load_segment("./"+imname+"_LayGrayResized.png", self.image_size)


			with torch.no_grad():
				img = wct2.transfer(content, style, content_segment, style_segment, 1)
			save_image(img.clamp_(0, 1), "./"+imname+"_manipulated.png", padding=0)

		elif(self.selected_stylemethod == 1):
			# Load model
			#Fast Photo Style Transfer
			p_wct = PhotoWCT()
			try:
	    			p_wct.load_state_dict(torch.load('./fastphoto/PhotoWCTModels/photo_wct.pth'))
			except:
				print("Fail to load PhotoWCT models. PhotoWCT submodule not updated?")
				exit()
			p_wct.cuda(0)

			process_stylization.stylization(p_wct=p_wct, content_image_path="./"+imname+"_original.png", style_image_path="./"+imname+"_G.png", content_seg_path="./"+imname+"_LayGray.png", style_seg_path="./"+imname+"_LayGrayResized.png", output_image_path="./"+imname+"_manipulated.png",
	    cuda=1)
			
		img_manipulated = Image.open("./"+imname+"_manipulated.png")
		self.imgtk = ImageTk.PhotoImage(image=img_manipulated)
		self.original_area.config(width=img_manipulated.size[0], height=img_manipulated.size[1])
		self.original_area.create_image(img_manipulated.size[0]/2 + 1,img_manipulated.size[1]/2 + 1,image=self.imgtk)
		self.original_area.after(50)
		self.original_area.update()
		self.show_button = ttk.Button(self.frame, text='Show Input Image', command=self.showCallBack)
		self.show_button.grid(row=4, column=5)
		self.lbl_oarea.configure(text = "Manipulated", font=("Helvetica", 18))

		if(vernum[0]==2):
			#photorealism
			#matlab -r  "try makeRealistic('$content_image', '$save_folder$name$png','$save_final$name$png');catch;end;quit"
			eng = matlab.engine.start_matlab()
			eng.cd(r'photorealism/', nargout=0)
			eng.makeRealistic("../"+imname+"_original.png", "../"+imname+"_manipulated.png", "../"+imname+"_manipulated.png", nargout=0)
			eng.quit()
			img_manipulated_post = Image.open("./"+imname+"_manipulated.png")
			self.imgtk = ImageTk.PhotoImage(image=img_manipulated_post)
			self.original_area.create_image(img_manipulated.size[0]/2 + 1,img_manipulated.size[1]/2 + 1,image=self.imgtk)
			self.original_area.after(50)
			self.original_area.update()

	def showCallBack(self):
		img_manipulated = Image.open("./"+imname+"_manipulated.png")
		if self.iforiginal:
			self.show_button.configure(text = "Show Input Image")
			self.lbl_oarea.configure(text = "Manipulated")
			self.imgtk = ImageTk.PhotoImage(image=img_manipulated)
			self.original_area.create_image(img_manipulated.size[0]/2 + 1,img_manipulated.size[1]/2 + 1,image=self.imgtk)
			self.original_area.after(50)
			self.original_area.update()
			self.iforiginal = False 
		else:
			self.show_button.configure(text = "Show Manipulated Image")
			self.lbl_oarea.configure(text = "Input")
			self.imgtk = ImageTk.PhotoImage(image=self.img_original)
			self.original_area.create_image(img_manipulated.size[0]/2 + 1,img_manipulated.size[1]/2 + 1,image=self.imgtk)
			self.original_area.after(50)
			self.original_area.update()
			self.iforiginal = True 



	def transform(self, X):
		npx = self.image_size
		X = [center_crop(x, npx) for x in X]
		npX = np.array(X)
		npX = npX.astype(float)
		return np.transpose(npX, (0, 3, 1, 2)) / 127.5 - 1


	def inverse_transform(self, X):
		npx = self.image_size
		X = (np.reshape(X, (-1, nc, npx, npx)).transpose(0, 2, 3, 1)+1.)/2.*255
		return X
	def b1down(self, event):
		self.b1 = "down" 


	def b1up(self, event):
		self.b1 = "up"
		self.xold = None 
		self.yold = None

		hexR = hex(self.colors[self.selected_object,0])
		hexG = hex(self.colors[self.selected_object, 1])
		hexB = hex(self.colors[self.selected_object, 2])
		hexR = hexR[2:]
		hexG = hexG[2:]
		hexB = hexB[2:]
		if len(hexR)==1:
			hexR = '0'+hexR
		if len(hexG)==1:
			hexG = '0'+hexG
		if len(hexB)==1:
			hexB = '0'+hexB

		backcolor = '#'+hexR + hexG + hexB
		event.widget.create_polygon(self.coords, fill=backcolor)
		self.drawn_object.append(self.coords[:])
		self.drawn_object_index.append(self.selected_object)
		#print(self.drawn_object_index)
		self.coords[:] = []
		self.drawObjects(event, self.drawn_object, self.drawn_object_index)

	def onselectStyleMethod(self, evt):
		# Note here that Tkinter passes an event object to onselect()
		w = evt.widget
		index = w.current()
		self.selected_stylemethod = index
		
	def onselect(self, evt):
		# Note here that Tkinter passes an event object to onselect()
		w = evt.widget
		index = int(w.curselection()[0])
		value = w.get(index)
		self.selected_object = index
		self.reset_lbobjects_colors()
		self.lbobjcurrent = -1
		#print('You selected item %d: "%s"' % (index, value))

	def reset_colors(self):
		"""Resets the colors of the items"""
		for ind,item in enumerate(self.attribute_lb.get(0, END)):
			self.attribute_lb.itemconfig(ind, {"bg": "white"})
			self.attribute_lb.itemconfig(ind, {"fg": "black"})

	def set_highlighted_item(self, index):
		"""Set the item at index with the highlighted colors"""
		self.attribute_lb.itemconfig(index, {"bg": "#eee8aa"})
		self.attribute_lb.itemconfig(index, {"fg": "black"})

	def mouseonAttribute(self, event):
		index = self.attribute_lb.index("@%s,%s" % (event.x, event.y))
		if self.lbcurrent != -1 and self.lbcurrent != index:
			self.reset_colors()
			self.set_highlighted_item(index)
		elif self.lbcurrent == -1:
			self.set_highlighted_item(index)
			self.lbcurrent = index

	def on_leave(self, event):
		self.reset_colors()
		self.lbcurrent = -1

	def reset_lbobjects_colors(self):
		"""Resets the colors of the items"""
		for ind,item in enumerate(self.lb_objects.get(0, END)):
			self.lb_objects.itemconfig(ind, {"bg": "white"})
			self.lb_objects.itemconfig(ind, {"fg": "black"})

	def set_highlighted_object_item(self, index):
		"""Set the item at index with the highlighted colors"""
		self.lb_objects.itemconfig(index, {"bg": "#eee8aa"})
		self.lb_objects.itemconfig(index, {"fg": "black"})

	def mouseonObjects(self, event):
		index = self.lb_objects.index("@%s,%s" % (event.x, event.y))
		if self.lbobjcurrent != -1 and self.lbobjcurrent != index:
			self.reset_lbobjects_colors()
			self.set_highlighted_object_item(index)
		elif self.lbobjcurrent == -1:
			self.set_highlighted_object_item(index)
			self.lbobjcurrent = index

	def on_leave_objects(self, event):
		self.reset_lbobjects_colors()
		self.lbobjcurrent = -1

	def onselectAttribute(self, evt):
		w = evt.widget
		index = int(w.curselection()[0])
		value = w.get(index)
		self.selected_attribute = index
		self.reset_colors()
		self.lbcurrent = -1
		#print('You selected item %d: "%s"' % (index, value))


	def drawObjects(self, event,objectlist,objectIndex):
		self.drawing_area.delete("line")
		count = 0
		imgDraw = ImageDraw.Draw(self.img_original_color_resized)
		imgDrawGray = ImageDraw.Draw(self.img_original_gray_resized)
		#print(objectlist)
		for obj in objectlist:

			hexR = hex(self.colors[objectIndex[count], 0])
			hexG = hex(self.colors[objectIndex[count], 1])
			hexB = hex(self.colors[objectIndex[count], 2])
			hexR = hexR[2:]
			hexG = hexG[2:]
			hexB = hexB[2:]
			if len(hexR) == 1:
			    hexR = '0' + hexR
			if len(hexG) == 1:
			    hexG = '0' + hexG
			if len(hexB) == 1:
			    hexB = '0' + hexB

			backcolor = '#' + hexR + hexG + hexB
			event.widget.create_polygon(obj, fill=backcolor)
			imgDraw.polygon(obj, fill=(self.colors[objectIndex[count], 0], self.colors[objectIndex[count], 1],self.colors[objectIndex[count], 2], 255))
			imgDrawGray.polygon(obj, fill=(objectIndex[count]+1))
			count = count + 1
			self.img_original_color_resized.save('./'+imname+'_LayColorResized.png',"PNG")
			self.img_original_gray_resized.save('./'+imname+'_LayGrayResized.png', "PNG")
			self.segmentBinary = self.binaryCodedImage(self.img_original_gray_resized)
			self.objectcategories = np.reshape(np.array(list(self.img_original_gray_resized.getdata())), (self.image_size, self.image_size))

	def motion(self, event):

		if event.x >= self.image_size:
			event.x = self.image_size
		if event.y >= self.image_size:
			event.y = self.image_size
		if event.x <= 1:
			event.x = 1
		if event.y <= 1:
			event.y = 1
		self.drawn_object=[]
		self.drawn_object_index = []
		if self.b1 == "down"  and event.x >= 1 and event.y >= 1:
			if self.xold is not None and self.yold is not None:
			    	event.widget.create_line(self.xold,self.yold,event.x,event.y,smooth=TRUE, tag="line" )

			self.xold = event.x
			self.yold = event.y
			self.coords.append((event.x - 1, event.y - 1))
			coords_start=self.coords[0]
			#print(coords_start)
			val = np.power((coords_start[0]-self.xold) , 2) + np.power((coords_start[1]-self.yold) , 2)
			#print('val = '+ str(val))
			dist = np.sqrt(val)
			#print('dist = '+ str(dist))
			if dist < 3.0 and len(self.coords) > 5:
				hexR = hex(self.colors[self.selected_object, 0])
				hexG = hex(self.colors[self.selected_object, 1])
				hexB = hex(self.colors[self.selected_object, 2])
				hexR = hexR[2:]
				hexG = hexG[2:]
				hexB = hexB[2:]
				if len(hexR) == 1:
					hexR = '0' + hexR
				if len(hexG) == 1:
					hexG = '0' + hexG
				if len(hexB) == 1:
					hexB = '0' + hexB

				backcolor = '#' + hexR + hexG + hexB
				event.widget.create_polygon(self.coords, outline=None ,fill=backcolor, width=0)
				self.drawn_object.append(self.coords[:])
				self.drawn_object_index.append(selected_object)
				self.coords[:] = []
				#print(self.drawn_object_index)

	def binaryCodedImage(self,catImage):
		img_np = np.array(list(catImage.getdata()))
		binaryim = np.zeros((self.image_size*self.image_size, 8))
		binaryim = np.zeros((catImage.size[0]*catImage.size[1], 8))
		for i in range(150):
			a = np.where(img_np==(i + 1))[0]
			for j in a:
				binaryim[j, :] = self.binarycodes[i, :]

		return binaryim.reshape(catImage.size[1], catImage.size[0], 8)

	def openNewImage(self):

		filename =  filedialog.askopenfilename(initialdir = "./",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
		#print(filename)

		num_val = -1

		list_test = [{'fpath_img': filename}]
		dataset_val = TestDataset(
		    list_test, opt, max_sample=num_val)
		loader_val = torchdata.DataLoader(
		    dataset_val,
		    batch_size=1,
		    shuffle=False,
		    collate_fn=user_scattered_collate,
		    num_workers=5,
		    drop_last=True)



		for i, batch_data in enumerate(loader_val):
			# process data
			batch_data = batch_data[0]
			segSize = (batch_data['img_ori'].shape[0],
			batch_data['img_ori'].shape[1])
			img_resized_list = batch_data['img_data']

			with torch.no_grad():
				pred = torch.zeros(1, 150, segSize[0], segSize[1])

				for img in img_resized_list:
					feed_dict = batch_data.copy()
					feed_dict['img_data'] = img
					del feed_dict['img_ori']
					del feed_dict['info']
					opt.gpu_id = 0
					feed_dict = async_copy_to(feed_dict, opt.gpu_id)

					# forward pass
					pred_tmp = self.segmentation_module(feed_dict, segSize=segSize)
					pred = pred + pred_tmp.cpu() / len(opt.imgSize)

				_, preds = torch.max(pred, dim=1)
				preds = as_numpy(preds.squeeze(0))
				where_are_NaNs = np.isnan(preds)
				preds[where_are_NaNs] = 0

						
	
					


		self.imtk = None
		self.iforiginal = False
		self.img = Image.open(filename)
		preds = preds.astype('uint8') + 1

		self.imgGray = Image.fromarray(preds)
		self.img_original = Image.open(filename)
		self.img_original, self.img_original_color, self.img_original_gray = self.transformImage2(self.img_original, self.imgGray)
		self.img_original_resized, self.img_original_color_resized, self.img_original_gray_resized = self.transformImage(self.img_original, self.imgGray)
		self.img_original_color.save('./' + imname + '_LayColor.png', "PNG")
		self.imgGray.save('./' + imname + '_LayGray.png', "PNG")
		self.img_original.save('./' + imname + '_original.png', "PNG")

		self.img_original_color_resized.save('./' + imname + '_LayColorResized.png', "PNG")
		self.img_original_gray_resized.save('./' + imname + '_LayGrayResized.png', "PNG")
		self.img_original_resized.save('./' + imname + '_originalResized.png', "PNG")

		self.segmentBinary = self.binaryCodedImage(self.img_original_gray_resized)
		self.objectcategories = np.reshape(np.array(list(self.img_original_gray_resized.getdata())), (self.img_original_gray_resized.size[0], self.img_original_gray_resized.size[1]))
		cat_np = self.objectcategories
		cat = torch.from_numpy(cat_np).float()
		category.resize_as_(cat.cuda()).copy_(cat)
		self.noise = self.init_z(category, 1)
		self.imseg = ImageTk.PhotoImage(image=self.img_original_color_resized)
		self.imorg = ImageTk.PhotoImage(image=self.img_original)
		self.lbl_oarea = Label(self.frame, text="Input", font=("Helvetica", 18), bg="gray")
		self.lbl_oarea.grid(row = 0, column = 5)
		self.original_area.destroy()
		self.original_area = Canvas(self.frame, width=self.img_original.size[0], height=self.img_original.size[1])
		self.original_area.grid(row=1, column=5)
		self.original_area.configure(background='white')

		self.drawing_area.create_image(self.image_size / 2 + 1, self.image_size / 2 + 1, image=self.imseg)

		self.original_area.create_image(self.img_original.size[0] / 2 + 1, self.img_original.size[1] / 2 + 1, image=self.imorg)
		self.original_area.config(width=self.img_original.size[0], height=self.img_original.size[1])



		self.canvas.create_window(0, 0, anchor='nw', window=self.frame)
		self.scroll_x.destroy()
		self.scroll_x = Scrollbar(self.root, orient="horizontal", command=self.canvas.xview)
		self.scroll_x.pack( fill=BOTH, side='top')
		self.canvas.update_idletasks()

		self.canvas.configure(scrollregion=self.canvas.bbox('all'), 
				 xscrollcommand=self.scroll_x.set)		 
		self.canvas.pack(fill='both', expand=True, side='top')

		self.canvas.configure(scrollregion=self.canvas.bbox('all'), 
				 xscrollcommand=self.scroll_x.set)
		self.root.mainloop()
	



if __name__ == "__main__":
    demo = DrawingWindow(imcolor_path, imoriginal_path, imgray_path, image_size)
