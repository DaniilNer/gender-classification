import json
import sys
from torchvision.transforms import ToTensor, Normalize
import torch
import torch.nn as nn
from torchvision.models import vgg13

import numpy as np
import numpy.testing as npt

from PIL import Image, ImageEnhance, ImageChops
import os
import re
import random


class Dataset(object):
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __add__(self, other):
        return ConcatDataset([self, other])
      
def sorted_aphanumeric_test(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(data, key=alphanum_key)

class PhotosDataset_test(Dataset):
    def __init__(self, images_dir, target_dir=None, transforms=None):
        self.masks = None
        self.dir = images_dir
        self.target_dir = target_dir
        self.transforms = transforms
        
        
        self.images = sorted_aphanumeric_test(os.listdir(images_dir))
        random.shuffle(self.images)
        
    def __len__(self):
        return len(os.listdir(self.dir))
    
    def __getitem__(self, idx):
        img = Image.open(self.dir+self.images[idx])
        if self.transforms:
            for trans in self.transforms:
              img = trans(img)
        return img


class HorizontalFlip(object):
    def __init__(self, prob = 1):
        self.prob = prob
    def __call__(self, img, mask=None):
        self.method = np.random.choice(2, 1, p =[1-self.prob,self.prob])[0]
        if self.method:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if mask:
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
                return img, mask
        return img
    
class cut(object):
    def __init__(self, prob = 1, output_size = None):
        self.prob = prob
        self.output_size = output_size
        
    def __call__(self, img, mask=None):
        self.method = np.random.choice(2, 1, p =[1-self.prob,self.prob])[0]
        if self.method:
            w, h = img.size
            if not self.output_size:
                new_w = int(np.random.normal(w/1.5, 30, 1)[0])
                new_h = int(np.random.normal(h/1.5, 30, 1)[0])

            else:
                new_w, new_h = self.output_size
            left = 0
            top = 0
            right = new_w
            bottom = new_h
            
            t = h - new_h
            l = w - new_w
            if t <= 0 and l<= 0:
                img = img.resize((new_w, new_h), Image.ANTIALIAS)
                if mask:
                    mask = mask.resize((new_w, new_h), Image.ANTIALIAS)
            elif t <= 0:
                img = img.resize((w, new_h), Image.ANTIALIAS)
                left = np.random.randint(0, l)
                right = left + new_w
            elif l <= 0:
                img = img.resize((new_w, h), Image.ANTIALIAS)
                top = np.random.randint(0, t)
                bottom = top + new_h
            else:
                top = np.random.randint(t)
                left = np.random.randint(l)
                bottom = top + new_h
                right = left + new_w
            img = img.crop((left, top, right, bottom))
            img = img.resize((224, 224), Image.ANTIALIAS)
            if mask:
                mask = mask.crop((left, top, right, bottom))
                mask = mask.resize((224, 224), Image.ANTIALIAS)
                return img, mask
            else: 
                return img
        if mask:
            return img, mask
        else: 
            img = img.resize((224, 224), Image.ANTIALIAS)
            return img

class brightness(object):
    def __init__(self, prob = 1):
        self.prob = prob
    def __call__(self, img):
        self.method = np.random.choice(2, 1, p =[1-self.prob,self.prob])[0]
        if self.method:
            br = np.random.uniform(0.3, 2.2, 1)[0]
            enh = ImageEnhance.Brightness(img)
            img = enh.enhance(br)
            return img
        return img
    
class background(object):
    def __init__(self, prob = 1, path = '/content/gdrive/My Drive/prak/bg_data'):
        self.prob = prob
        self.path = path
        self.bg = os.listdir(self.path)
    def __call__(self, img, mask):
        self.method = np.random.choice(2, 1, p =[1-self.prob,self.prob])[0]
        if self.method:
            idx = np.random.randint(len(self.bg))
            bg_i = Image.open(os.path.join(self.path, self.bg[idx]))
            bg_i = cut(output_size = (img.size()[2], img.size()[1]))(bg_i)
            
            bg_i = ToTensor()(bg_i)
            img1 = bg_i * abs(mask-1)
            img2 = img * mask
            img = img1 + img2
        return img, mask
    
class ToTensor1(object):
    def __call__(self, img, mask=None):
        img = ToTensor()(img)
        if mask:
          mask = ToTensor()(mask)
          return img, mask
        return img

class VGG(nn.Module):
    def __init__(self, features, output_dim):
        super().__init__()
        
        self.features = features
        
        self.avgpool = nn.AdaptiveAvgPool2d(7)
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(0.5),
            nn.Linear(4096, 2622),
            nn.ReLU(inplace = True),
            nn.Dropout(0.5),
            nn.Linear(2622, output_dim),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        h = x.view(x.shape[0], -1)
        x = self.classifier(h)
        return x, h

def get_vgg_layers(config, batch_norm):
    
    layers = []
    in_channels = 3
    
    for c in config:
        assert c == 'M' or isinstance(c, int)
        if c == 'M':
            layers += [nn.MaxPool2d(kernel_size = 2)]
        else:
            conv2d = nn.Conv2d(in_channels, c, kernel_size = 3, padding = 1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(c), nn.ReLU(inplace = True)]
            else:
                layers += [conv2d, nn.ReLU(inplace = True)]
            in_channels = c
            
    return nn.Sequential(*layers)

def pred(net, testloader, device='cpu'):
    net = net.eval()
    device = torch.device(device)
    p = []
    with torch.no_grad():
        for data in testloader:
            images = data
            images = images
            
            images = images.to(device)
            outputs, _ = net(images)
            outputs = outputs.to(device)
            _, prediction = torch.max(outputs.data, 1)
            p.append(prediction)
    
    return p

def main(fold):
    DATA_PATH = fold#'/content/gdrive/My Drive/test/'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vgg11_config = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
    test_dataset = PhotosDataset_test(
        images_dir=DATA_PATH + 'tt/',
        transforms= [cut(0.), ToTensor1(), Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])]
    )
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64)

    vgg11_layers = get_vgg_layers(vgg11_config, batch_norm = True)
    OUTPUT_DIM = 2
    model = VGG(vgg11_layers, OUTPUT_DIM)
    model = model.to(device)
    model.load_state_dict(torch.load('nn22_2.model'))
    p = pred(model, test_data_loader, device)
    p = p[0].tolist()
    ps = ["male" if x==1 else "female" for x in p]
    d = dict(zip(test_dataset.images, ps))
    with open('process_results.json', 'w') as outfile:
        json.dump(d, outfile)

if __name__ == "__main__":
    main(sys.argv[1])
