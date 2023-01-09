from collections import OrderedDict
from datetime import datetime
from pathlib import Path
import random
from re import L

import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.utils.torch as fotorch
import fiftyone.utils.annotations as foua
import fiftyone.utils.patches as foup
from sklearn import cluster

import torch
import torchvision.datasets as dset
import torch.optim as optim
import torch.nn as nn
import torchvision as tv
from torchvision import transforms as tf
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms import functional as tfunc
from torch.utils.data import DataLoader
import torchvision.utils as vutils

import torchviz

from PIL import Image

import matplotlib.pyplot as plt

import numpy as np
from sklearn.cluster import KMeans, OPTICS, DBSCAN, Birch


COCO_2017_DATASET_DIR = Path("./data/coco_2017/")

def draw_image(image, fname=None, show=False):
    if isinstance(image, torch.Tensor):
        if str(image.device) == 'cuda:0':
            image = image.detach().cpu()
        image = image.squeeze().numpy()
    plt.figure(figsize=(8,8))
    plt.axis("off")
    if len(image.shape) == 2:
        plt.imshow(image, interpolation="none")
    else:
        image = np.transpose(image, (1,2,0))
        plt.imshow(image, interpolation="none")
    if show:
        plt.show()

    if fname:
        plt.savefig(fname)
    plt.close()
    
def draw_layers(data, fname=None, show=False):
    if isinstance(data, torch.Tensor):
        if str(data.device) == 'cuda:0':
            data = data.detach().cpu()
        data = data.squeeze().numpy()
    if len(data.shape) == 2:
        return draw_image(data)
    data_layers = [tfunc.to_tensor(d) for d in data]
    grid_image = vutils.make_grid(data_layers, nrow=int(len(data_layers)**0.5), padding=0, pad_value=0.5, normalize=True).cpu()
    draw_image(grid_image, fname, show)


class FPNStage(nn.Module):
    def __init__(self, fpn_dim, bbone_dim):
        super().__init__()
        self.lat = nn.Conv2d(bbone_dim, fpn_dim, kernel_size=1)
        self.top = nn.ConvTranspose2d(fpn_dim, fpn_dim, kernel_size=4, stride=2, padding=1)
        self.aa  = nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, stride=1, padding=1)
        
    def forward(self, bbone_activation, prev_fpn_stage):
        lat_out = self.lat(bbone_activation)
        top_out = self.top(prev_fpn_stage)
        
        if not lat_out.shape == top_out.shape:
            top_out = nn.UpsamplingNearest2d(size=lat_out.shape[2:])(top_out)          
            
        final_out = self.aa(lat_out + top_out)
        return final_out
        

class FeatureExtractor(nn.Module):
    def __init__(self, fpn_dim=256):
        super().__init__()
        self.num_fpn_stages = 4
        self.fpn_dim = fpn_dim
        self.fpn = tv.ops.FeaturePyramidNetwork(
            in_channels_list=[256, 512, 1024, 2048], out_channels=fpn_dim
        )
        self.feat_extender = nn.Sequential(
            nn.ConvTranspose2d(fpn_dim, fpn_dim, 4, 2, 1),
            nn.BatchNorm2d(fpn_dim),
            nn.ConvTranspose2d(fpn_dim, fpn_dim, 4, 2, 1),
            nn.BatchNorm2d(fpn_dim),
            nn.ReLU(inplace=True)
        )
        
        self.activation = {}
        self.resnet50_backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        # self.resnet50_backbone.layer1.register_forward_hook(self.get_activation('conv2'))
        # self.resnet50_backbone.layer2.register_forward_hook(self.get_activation('conv3'))
        # self.resnet50_backbone.layer3.register_forward_hook(self.get_activation('conv4'))
        # self.resnet50_backbone.layer4.register_forward_hook(self.get_activation('conv5'))
        
        # self.fpn_stage_1 = nn.Conv2d(2048, self.fpn_dim, kernel_size=1)
        # self.fpn_stage_2 = FPNStage(self.fpn_dim, 1024)
        # self.fpn_stage_3 = FPNStage(self.fpn_dim, 512)
        # self.fpn_stage_4 = FPNStage(self.fpn_dim, 256)
        
    def forward(self, input):
        out = self.resnet50_backbone.conv1(input)
        out = self.resnet50_backbone.bn1(out)
        out = self.resnet50_backbone.relu(out)
        out = self.resnet50_backbone.maxpool(out)
        
        feat0 = self.resnet50_backbone.layer1(out)
        feat1 = self.resnet50_backbone.layer2(feat0)
        feat2 = self.resnet50_backbone.layer3(feat1)
        feat3 = self.resnet50_backbone.layer4(feat2)

        feats = OrderedDict()
        feats["0"] = feat0
        feats["1"] = feat1
        feats["2"] = feat2
        feats["3"] = feat3

        out_feats = self.fpn(feats)
        # out_feats = self.feat_extender(out_feats["0"])
        # dense_feats = out_feats["3"]
        # flat_feats = self.resnet.avgpool(dense_feats)
        # flat_feats = torch.flatten(flat_feats, 1)


        # fpn_stage_1_output = self.fpn_stage_1(self.activation['conv5'])
        # fpn_stage_2_output = self.fpn_stage_2(self.activation['conv4'], fpn_stage_1_output)
        # fpn_stage_3_output = self.fpn_stage_3(self.activation['conv3'], fpn_stage_2_output)
        # fpn_stage_4_output = self.fpn_stage_4(self.activation['conv2'], fpn_stage_3_output)

        # return out_feats
        return out_feats["0"]
        
    def get_activation(self, name):
        def hook(model, input, output):
            self.activation[name] = output.detach()
        return hook 


class Projector(nn.Module):
    def __init__(self, fpn_dim=256):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(fpn_dim, fpn_dim, 1),
            nn.ReLU(inplace=False),
            nn.Linear(fpn_dim, fpn_dim, 1),
            nn.ReLU(inplace=False)
        )
        
    def forward(self, input):
        return self.main(input)


class Predictor(nn.Module):
    def __init__(self, fpn_dim=256):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(fpn_dim, fpn_dim, 1),
            nn.ReLU(inplace=False),
            nn.Linear(fpn_dim, fpn_dim, 1),
            nn.ReLU(inplace=False)
        )
        
    def forward(self, input):
        return self.main(input)


class TauModel(nn.Module):
    def __init__(self, fpn_dim=256):
        super().__init__()
        self.f = FeatureExtractor(fpn_dim=fpn_dim)
        self.g = Projector(fpn_dim=fpn_dim)
        self.q = Predictor(fpn_dim=fpn_dim)
        
    def forward(self, input):
        h = tfunc.resize(self.f(input), (448, 448), tf.InterpolationMode.BILINEAR)
        z = self.g(h.transpose(1,3))
        p = self.q(z)
        return h, z, p
        
class ThetaXiModel(nn.Module):
    def __init__(self, fpn_dim=256):
        super().__init__()
        self.f = FeatureExtractor(fpn_dim=fpn_dim)
        self.g = Projector(fpn_dim=fpn_dim)
        self.q = Predictor(fpn_dim=fpn_dim)
        
    def forward(self, input):
        h = tfunc.resize(self.f(input), (224, 224), tf.InterpolationMode.BILINEAR)
        z = self.g(h)
        p = self.q(z)
        return h, z, p
    

class ViewGenerator(nn.Module):
    """
    nn.Module sub-class which, when called on an image, generates three views of the image, v0, v1, and v2. v1
    and v2 are generated first, and then v0 is generated from their bounding box. The class also has a
    `reverse` method which, given an image the size of v0, produces the equivalent v1 and v2 crops from it.
    """
    SCALE_RANGE = (0.08, 1.0)
    RATIO_RANGE = (0.75, 1.33333333333)

    FLIP_PROB = 0.5

    COLOR_JITTER_PROB = 0.8
    COLOR_OPERATIONS = ["brightness", "contrast", "saturation", "hue"]
    BRIGHTNESS_MAX = 0.4
    CONTRAST_MAX = 0.4
    SATURATION_MAX = 0.2
    HUE_MAX = 0.1

    GRAY_PROB = 0.2

    v1_BLUR_PROB = 1.0
    v2_BLUR_PROB = 0.1

    v1_SOLAR_PROB = 0.0
    v2_SOLAR_PROB = 0.2

    v0_SHAPE = (256, 256)
    v1_SHAPE = v2_SHAPE = (128, 128)
    INTERPOLATION = tf.InterpolationMode.BILINEAR

    def __init__(self, image):
        super().__init__()
        self.crop_v1= tf.RandomResizedCrop.get_params(image, self.SCALE_RANGE, self.RATIO_RANGE)
        self.crop_v2 = tf.RandomResizedCrop.get_params(image, self.SCALE_RANGE, self.RATIO_RANGE)
        
        flip_v1 = random.random() < self.FLIP_PROB
        flip_v2 = random.random() < self.FLIP_PROB
        
        gray_v1 = random.random() < self.GRAY_PROB
        gray_v2 = random.random() < self.GRAY_PROB
        
        blur_v1 = random.random() < self.v1_BLUR_PROB
        blur_v2 = random.random() < self.v2_BLUR_PROB
        
        solar_v1 = random.random() < self.v1_SOLAR_PROB
        solar_v2 = random.random() < self.v2_SOLAR_PROB
        
        if random.random() < self.COLOR_JITTER_PROB: 
            color_params = tf.ColorJitter.get_params(
                [max(0, 1 - self.BRIGHTNESS_MAX), 1 + self.BRIGHTNESS_MAX],
                [max(0, 1 - self.CONTRAST_MAX), 1 + self.CONTRAST_MAX],
                [max(0, 1 - self.SATURATION_MAX), 1 + self.SATURATION_MAX],
                [-self.HUE_MAX, self.HUE_MAX]
            )
            order = color_params[0]
            color_params = color_params[1:]
            jitter_v1 = [(self.COLOR_OPERATIONS[i], color_params[i]) for i in order]
        else:
            jitter_v1 = None
        if random.random() < self.COLOR_JITTER_PROB: 
            color_params = tf.ColorJitter.get_params(
                [max(0, 1 - self.BRIGHTNESS_MAX), 1 + self.BRIGHTNESS_MAX],
                [max(0, 1 - self.CONTRAST_MAX), 1 + self.CONTRAST_MAX],
                [max(0, 1 - self.SATURATION_MAX), 1 + self.SATURATION_MAX],
                [-self.HUE_MAX, self.HUE_MAX]
            )
            order = color_params[0]
            color_params = color_params[1:]
            jitter_v2 = [(self.COLOR_OPERATIONS[i], color_params[i]) for i in order]
        else:
            jitter_v2 = None
        
        self.v1_params = (self.crop_v1, flip_v1, jitter_v1, gray_v1, blur_v1, solar_v1, self.v1_SHAPE)
        self.v2_params = (self.crop_v2, flip_v2, jitter_v2, gray_v2, blur_v2, solar_v2, self.v2_SHAPE)
        
        self.v1_proportional_crop = None
        self.v2_proportional_crop = None
        
    def __call__(self, img):
        v1 = self._generate_sub_view(img, self.v1_params)
        v2 = self._generate_sub_view(img, self.v2_params)
        v0 = self._generate_whole_view(img, self.crop_v1, self.crop_v2)
        
        return v0, v1, v2
    
    def reverse(self, image):
        image_height, image_width = image.shape[-2:]
        
        v1_top_scaled = round(self.v1_proportional_crop[0] * image_height)
        v1_left_scaled = round(self.v1_proportional_crop[1] * image_width)
        v1_height_scaled = round(self.v1_proportional_crop[2] * image_height)
        v1_width_scaled = round(self.v1_proportional_crop[3] * image_width)
        
        v2_top_scaled = round(self.v2_proportional_crop[0] * image_height)
        v2_left_scaled = round(self.v2_proportional_crop[1] * image_width)
        v2_height_scaled = round(self.v2_proportional_crop[2] * image_height)
        v2_width_scaled = round(self.v2_proportional_crop[3] * image_width)
        
        v1_scaled = tfunc.resized_crop(image, v1_top_scaled, v1_left_scaled, v1_height_scaled, v1_width_scaled, self.v1_SHAPE, tf.InterpolationMode.BILINEAR)
        v2_scaled = tfunc.resized_crop(image, v2_top_scaled, v2_left_scaled, v2_height_scaled, v2_width_scaled, self.v2_SHAPE, tf.InterpolationMode.BILINEAR)
        
        if self.v1_params[1]: v1_scaled = tfunc.hflip(v1_scaled)
        if self.v2_params[1]: v2_scaled = tfunc.hflip(v2_scaled)
        
        return v1_scaled, v2_scaled
        
    def _generate_whole_view(self, image, crop_v1, crop_v2):
        v1_top, v1_left, v1_height, v1_width = crop_v1
        v2_top, v2_left, v2_height, v2_width = crop_v2
        v1_bot = v1_top + v1_height
        v1_right = v1_left + v1_width
        v2_bot = v2_top + v2_height
        v2_right = v2_left + v2_width
        
        v0_top = min(v1_top, v2_top)
        v0_left = min(v1_left, v2_left)
        v0_bot = max(v1_bot, v2_bot)    
        v0_right = max(v1_right, v2_right)
        v0_height = v0_bot - v0_top
        v0_width = v0_right - v0_left
        
        self.v0_crop = (v0_top, v0_left, v0_height, v0_width)
        
        v1_proportional_top = (v1_top - v0_top)/v0_height
        v1_proportional_left = (v1_left - v0_left)/v0_width
        v1_proportional_height = v1_height/v0_height
        v1_proportional_width = v1_width/v0_width
        self.v1_proportional_crop = (v1_proportional_top, v1_proportional_left, v1_proportional_height, v1_proportional_width)
        
        v2_proportional_top = (v2_top - v0_top)/v0_height
        v2_proportional_left = (v2_left - v0_left)/v0_width
        v2_proportional_height = v2_height/v0_height
        v2_proportional_width = v2_width/v0_width
        self.v2_proportional_crop = (v2_proportional_top, v2_proportional_left, v2_proportional_height, v2_proportional_width)
        
        return tfunc.resized_crop(image, v0_top, v0_left, v0_height, v0_width, self.v0_SHAPE, tf.InterpolationMode.BILINEAR)
    
    def _generate_sub_view(self, image, params):
        crop, flip, jitter, gray, blur, solar, shape = params
        
        t, l, h, w = crop
        output = tfunc.resized_crop(image, t, l, h, w, shape, self.INTERPOLATION)
        if flip: output = tfunc.hflip(output)
        if jitter is not None:
            for param, value in jitter:
                if param == "brightness": output = tfunc.adjust_brightness(output, value)
                elif param == "contrast": output = tfunc.adjust_contrast(output, value)
                elif param == "hue": output = tfunc.adjust_hue(output, value)
                elif param == "saturation": output = tfunc.adjust_saturation(output, value)
        if gray: output = tfunc.rgb_to_grayscale(output, 3)
        if blur: output = tfunc.gaussian_blur(output, 23, (0.1, 2.0))
        if solar: output = tfunc.solarize(output, 0.5)

        return output


def make_dataloaders():
    if not COCO_2017_DATASET_DIR.exists():
        foz.download_zoo_dataset("coco-2017", dataset_dir=str(COCO_2017_DATASET_DIR))
    
    training_dataset = dset.CocoDetection(
        root=str(COCO_2017_DATASET_DIR.joinpath("train/data")),
        annFile=str(COCO_2017_DATASET_DIR.joinpath("train/labels.json")),
        transform=tf.Compose([tf.ToTensor()])
        )
    training_dataloader = DataLoader(training_dataset, batch_size=1, shuffle=True)

    testing_dataset = dset.CocoDetection(
        root=str(COCO_2017_DATASET_DIR.joinpath("test/data")),
        annFile=str(COCO_2017_DATASET_DIR.joinpath("test/labels.json")),
        transform=tf.Compose([tf.ToTensor()])
        )
    testing_dataloader = DataLoader(testing_dataset, batch_size=1, shuffle=True)

    validation_dataset = dset.CocoDetection(
        root=str(COCO_2017_DATASET_DIR.joinpath("validation/data")),
        annFile=str(COCO_2017_DATASET_DIR.joinpath("train/labels.json")),
        transform=tf.Compose([tf.ToTensor()])
        )
    validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=True)

    # return testing_dataloader, testing_dataloader, testing_dataloader
    return training_dataloader, testing_dataloader, validation_dataloader


class ODIN(object):
    def __init__(self, train_dataloader, test_dataloader, val_dataloader, 
                fpn_dim=256, lr_tau=0.2, lr_theta=0.5, lr_xi=0.2, clusters=8,
                device='cuda:0'):
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.val_dataloader = val_dataloader

        self.device = device

        self.num_clusters = clusters
        self.clusterer = None

        self.f_tau = FeatureExtractor(fpn_dim).to(self.device)
        self.g_tau = Projector(fpn_dim).to(self.device)

        self.f_theta = FeatureExtractor(fpn_dim).to(self.device)
        self.g_theta = Projector(fpn_dim).to(self.device)
        self.q_theta = Predictor(fpn_dim).to(self.device)

        self.f_xi = FeatureExtractor(fpn_dim).to(self.device)
        self.g_xi = Projector(fpn_dim).to(self.device)

        self.f_theta_optim = optim.SGD(self.f_theta.parameters(), lr=lr_theta)
        self.g_theta_optim = optim.SGD(self.g_theta.parameters(), lr=lr_theta)
        self.q_theta_optim = optim.SGD(self.q_theta.parameters(), lr=lr_theta)

        self.lr_tau = lr_tau
        self.lr_theta = lr_theta
        self.lr_xi = lr_xi

    def set_num_clusters(self, new_num):
        self.num_clusters = new_num

    # def set_clusterer(self, clusterer):
    #     self.clusterer = clusterer

    def generate_masks(self, feature_map, view_gen=None, clusterer=None):
        if clusterer is None:
            clusterer = self.clusterer
        feature_map_np = np.transpose(feature_map.detach().cpu().squeeze(),(1,2,0))
        
        original_shape = feature_map_np.shape
        flat_shape = (original_shape[0]*original_shape[1], original_shape[2])
        feature_map_np_flat = feature_map_np.reshape((flat_shape))
        
        mask_assignments_flat = clusterer.fit_predict(feature_map_np_flat)
        cluster_ids = set(mask_assignments_flat)
        if -1 in cluster_ids:
            cluster_ids.remove(-1)
        
        mask_assignments = tfunc.to_tensor(mask_assignments_flat.reshape(original_shape[:2])).squeeze()
        m0_layers = [torch.where(mask_assignments==c_id, 1., 0.).numpy() for c_id in sorted(cluster_ids)]
        m0_np = np.stack(m0_layers, 2)
        m0 = tfunc.to_tensor(m0_np).to(self.device)

        if view_gen:    
            m1_raw, m2_raw = view_gen.reverse(m0)
            m1 = m1_raw[torch.argwhere(m1_raw.sum(dim=(1,2))>0)].squeeze()
            m2 = m2_raw[torch.argwhere(m2_raw.sum(dim=(1,2))>0)].squeeze()
            m0 = m0.reshape((1,) + m0.shape)
            m1 = m1.reshape((1,) + m1.shape)
            m2 = m2.reshape((1,) + m2.shape)
        else:
            m1 = m2 = None
        
        return cluster_ids, mask_assignments, m0, m1, m2

    def single_mask_similarity(self, p_theta, z_xi, alpha):
        top = torch.dot(p_theta, z_xi)
        bot = torch.mul(torch.norm(p_theta, p=2), torch.norm(z_xi, p=2))
        return (1.0/alpha)*(top/bot)

    def feature_contrastive_loss(self, p_k_1_theta, z_k_2_xi, index, alpha=0.1):
        if index >= len(p_k_1_theta): return 0
        if index >= len(z_k_2_xi): return 0
        positive_similarity = self.single_mask_similarity(p_k_1_theta[index], z_k_2_xi[index], alpha)
        negative_similarities_sum = sum([self.single_mask_similarity(p_k_1_theta[index], zk2_xi, alpha) for i, zk2_xi in enumerate(z_k_2_xi) if i != index])
        bot = positive_similarity + negative_similarities_sum
        nll = -1*torch.log(positive_similarity/bot)
        return nll

    def total_contrastive_loss(self, p_k_1_theta, p_k_2_theta, z_k_1_xi, z_k_2_xi, alpha):
        cum_loss = torch.Tensor([0.]).to(self.device)
        num_masks_found = min([len(p_k_1_theta), len(p_k_2_theta), len(z_k_1_xi), len(z_k_2_xi)])
        if num_masks_found == 0:
            num_masks_found = 1e-7
        for mask_idx in range(num_masks_found):
            l_12_k = self.feature_contrastive_loss(p_k_1_theta, z_k_2_xi, mask_idx, alpha)
            l_21_k = self.feature_contrastive_loss(p_k_2_theta, z_k_1_xi, mask_idx, alpha)
            cum_loss += l_12_k + l_21_k
        return cum_loss/num_masks_found

    def within_mask_closeness_loss(self, masked_h):
        within_h_max = torch.where(masked_h != 0, masked_h, float("-Inf")).amax(dim=(2,3))
        within_h_min = torch.where(masked_h != 0, masked_h, float("Inf")).amin(dim=(2,3))
        within_h_range = torch.abs(within_h_max - within_h_min)
        average_range = within_h_range.mean()
        return average_range

    def run_tau_network(self, v0):
        h_0 = self.f_tau(v0)
        z_0 = self.g_tau(h_0.transpose(1,3)).transpose(3,1)
        return h_0, z_0

    def run_theta_network(self, v1, v2, m1, m2):
        h_1 = tfunc.resize(self.f_theta(v1), (128,128), tf.InterpolationMode.BILINEAR)
        h_2 = tfunc.resize(self.f_theta(v2), (128,128), tf.InterpolationMode.BILINEAR)
        # h_1 = self.f_theta(v1)
        # h_2 = self.f_theta(v2)

        masked_h_1 = torch.concat([torch.concat([torch.mul(m1_i, h_1) for m1_i in m1_j]) for m1_j in m1])
        masked_h_2 = torch.concat([torch.concat([torch.mul(m2_i, h_2) for m2_i in m2_j]) for m2_j in m2])

        sum_masked_h_1 = masked_h_1.sum(dim=(2,3))
        sum_masked_h_2 = masked_h_2.sum(dim=(2,3))

        m1_sums = m1.sum(dim=(2,3)).transpose(1,0)
        m2_sums = m2.sum(dim=(2,3)).transpose(1,0)

        h_k_1 = sum_masked_h_1/(m1_sums)
        h_k_2 = sum_masked_h_2/(m2_sums)

        z_k_1 = self.g_theta(h_k_1)
        z_k_2 = self.g_theta(h_k_2)

        p_k_1 = self.q_theta(z_k_1)
        p_k_2 = self.q_theta(z_k_2)

        return (h_1, h_2), (masked_h_1, masked_h_2), (h_k_1, h_k_2), (z_k_1, z_k_2), (p_k_1, p_k_2)

    def run_xi_network(self, v1, v2, m1, m2):
        h_1 = tfunc.resize(self.f_xi(v1), (128,128), tf.InterpolationMode.BILINEAR)
        h_2 = tfunc.resize(self.f_xi(v2), (128,128), tf.InterpolationMode.BILINEAR)
        # h_1 = self.f_xi(v1)
        # h_2 = self.f_xi(v2)

        masked_h_1 = torch.concat([torch.concat([torch.mul(m1_i, h_1) for m1_i in m1_j]) for m1_j in m1])
        masked_h_2 = torch.concat([torch.concat([torch.mul(m2_i, h_1) for m2_i in m2_j]) for m2_j in m2])

        sum_masked_h_1 = masked_h_1.sum(dim=(2,3))
        sum_masked_h_2 = masked_h_2.sum(dim=(2,3))

        m1_sums = m1.sum(dim=(2,3)).transpose(1,0)
        m2_sums = m2.sum(dim=(2,3)).transpose(1,0)

        h_k_1 = sum_masked_h_1/(m1_sums)
        h_k_2 = sum_masked_h_2/(m2_sums)

        z_k_1 = self.g_xi(h_k_1)
        z_k_2 = self.g_xi(h_k_2)

        return (h_1, h_2), (masked_h_1, masked_h_2), (h_k_1, h_k_2), (z_k_1, z_k_2)

    def byol_parameter_adjustment(self, param_zip):
        with torch.no_grad():
            for p_f_tau, p_f_theta, p_f_xi in param_zip:
                new_p_f_xi = (1-self.lr_xi)*p_f_xi + self.lr_xi*p_f_theta
                new_p_f_tau = (1-self.lr_tau)*p_f_tau + self.lr_tau*p_f_theta

                p_f_xi.copy_(new_p_f_xi)
                p_f_tau.copy_(new_p_f_tau)

    def save_parameters(self, save_dir:Path):
        torch.save(self.f_tau.state_dict(), save_dir.joinpath("f_tau.pth"))
        torch.save(self.g_tau.state_dict(), save_dir.joinpath("g_tau.pth"))

        torch.save(self.f_theta.state_dict(), save_dir.joinpath("f_theta.pth"))
        torch.save(self.g_theta.state_dict(), save_dir.joinpath("g_theta.pth"))
        torch.save(self.q_theta.state_dict(), save_dir.joinpath("q_theta.pth"))

        torch.save(self.f_xi.state_dict(), save_dir.joinpath("f_xi.pth"))
        torch.save(self.g_xi.state_dict(), save_dir.joinpath("g_xi.pth"))

        with save_dir.joinpath("f_tau_params_raw.txt").open("w") as params_raw_file:
            params_raw_file.write(str(list(self.f_tau.named_parameters())))

    def get_clusterer(self, feature_map, eps_coeff=1.0, num_clusters=4, clusterer_type="kmeans"):
        fmap_dists = torch.cdist(feature_map, feature_map, p=2)
        fmap_dist_mean = fmap_dists.mean().item()
        fmap_dist_std_dev = fmap_dists.std().item()
        fmap_dist_median = fmap_dists.median().item()

        fmap_norms = torch.norm(feature_map, p=2, dim=1)
        fmap_norms_mean = fmap_norms.mean().item()
        fmap_norms_std_dev = fmap_norms.std().item()
        fmap_norms_median = fmap_norms.median().item()

        epsilon = fmap_norms_median * eps_coeff

        clusterer_OPTICS = OPTICS(
            cluster_method="dbscan",
            min_samples=0.05, 
            eps=epsilon,
            n_jobs=4
            )
        clusterer_kmeans = KMeans(
            n_clusters=num_clusters,
            n_init=10, 
            max_iter=500,
            tol=0.0001,
            copy_x=False,
            algorithm='elkan'
            )

        if clusterer_type == "kmeans":
            return clusterer_kmeans
        elif clusterer_type == "optics":
            return clusterer_OPTICS
        elif clusterer_type == "both":
            return clusterer_kmeans, clusterer_OPTICS
        else:
            raise ValueError(clusterer_type)

    def run_networks_training_mode(self, v1, v2, m1, m2, loss_method="contrastive"):
        _, (masked_h1_theta, masked_h2_theta), __, ___, (pk1_theta, pk2_theta) = self.run_theta_network(v1, v2, m1, m2)
        _, __, ___, (zk1_xi, zk2_xi) = self.run_xi_network(v1, v2, m1, m2)
        constrastive_loss = self.total_contrastive_loss(pk1_theta, pk2_theta, zk1_xi, zk2_xi, 0.1)
        closeness_loss = (self.within_mask_closeness_loss(masked_h1_theta) + self.within_mask_closeness_loss(masked_h2_theta))/2.

        if loss_method == "both":
            loss = torch.max(constrastive_loss, closeness_loss)
        else:
            loss = constrastive_loss
        # loss = closeness_loss

        loss.backward()
        self.q_theta_optim.step()
        self.g_theta_optim.step()
        self.f_theta_optim.step()

        self.byol_parameter_adjustment(zip(self.f_tau.parameters(), self.f_theta.parameters(), self.f_xi.parameters()))
        self.byol_parameter_adjustment(zip(self.g_tau.parameters(), self.g_theta.parameters(), self.g_xi.parameters()))

        return loss, constrastive_loss, closeness_loss

    def run_training_iteration(self, input_tensor, eps_coeff, clusterer_type="kmeans", loss_method="contrastive"):
        view_gen = ViewGenerator(input_tensor)
        v0, v1, v2 = view_gen(input_tensor)

        h0, z0 = self.run_tau_network(v0)
        
        cluster_results = dict()
        total_loss = torch.Tensor([0.]).to(self.device)
        if clusterer_type in ["kmeans", "both"]:
            clusterer = self.get_clusterer(h0, eps_coeff, self.num_clusters, clusterer_type="kmeans")
            cluster_ids, masks, m0, m1, m2 = self.generate_masks(h0, view_gen, clusterer)
            loss, constrastive_loss, closeness_loss = self.run_networks_training_mode(v1,v2, m1, m2, loss_method)
            cluster_results["kmeans"] = (cluster_ids, masks, m0, m1, m2, loss, constrastive_loss, closeness_loss)
            total_loss += loss

        if clusterer_type in ["optics", "both"]:
            clusterer = self.get_clusterer(h0, eps_coeff, self.num_clusters, clusterer_type="optics")
            cluster_ids, masks, m0, m1, m2 = self.generate_masks(h0, view_gen, clusterer)
            loss, constrastive_loss, closeness_loss = self.run_networks_training_mode(v1,v2, m1, m2, loss_method)
            cluster_results["optics"] = (cluster_ids, masks, m0, m1, m2,  loss, constrastive_loss, closeness_loss)
            total_loss += loss

        return cluster_results, total_loss

    def train(self, iterations=1000, save_steps=25, clusterer_type="kmeans", loss_method="contrastive", session_id=None):
        if session_id is None:
            session_id = f"{datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M-%S')}_{self.num_clusters}"
        session_path = Path(f"./runs/{session_id}/")
        if session_path.exists():
            session_path = session_path.joinpath(f"{datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M-%S')}/")
        session_path.mkdir(exist_ok=True)
        losses_log_file = session_path.joinpath("losses.txt")
        losses_log_file.touch()
        losses = []

        demo_tensor = next(iter(self.test_dataloader))[0].to(self.device)
        consecutive_skips = 0

        self.eps_coeff = 1.0
        for i, input_raw in enumerate(self.train_dataloader):
            self.f_tau.train()
            self.g_tau.train()
            self.f_theta.train()
            self.g_theta.train()
            self.q_theta.train()
            self.f_xi.train()
            self.g_xi.train()

            self.f_theta_optim.zero_grad()
            self.g_theta_optim.zero_grad()
            self.q_theta_optim.zero_grad()

            input_tensor = input_raw[0].to(self.device)
            try:
                clusters_data, loss = self.run_training_iteration(input_tensor, self.eps_coeff, clusterer_type, loss_method)
                if "optics" in clusters_data:
                    if len(clusters_data["optics"][0]) == 0:
                        # There aren't any clusters, increase epsilon to make it easier for clusters to form
                        consecutive_skips += 1
                        self.eps_coeff += 0.1 * consecutive_skips
                        continue
                    if len(clusters_data["optics"][0]) == 1:
                        # There's only one cluster, decrease epsilon to make it easier for clusters to differentiate
                        self.eps_coeff -= 0.1
                else:
                    clusters_data["optics"] = [[]]
                if 'kmeans' not in clusters_data:
                    clusters_data["kmeans"] = [[]]
                consecutive_skips = 0
                loss_val = loss.item()
                losses.append(loss_val)
                past_10_variance = np.var(losses[-10:])

                losses_log_file.open('a').write(f"{i},{loss_val}\n")
                print(f"Iter {i} ({len(clusters_data['kmeans'][0])}, {len(clusters_data['optics'][0])}): {loss_val:.5f}, {past_10_variance:.8f}, {clusters_data['kmeans'][6].item():.5f}, {clusters_data['kmeans'][7].item():.5f}")
            except Exception as e:
                plt.plot(losses)
                plt.savefig(str(session_path.joinpath("loss_graph.png")))
                save_path = session_path.joinpath(f"{i}/")
                save_path.mkdir()
                self.inference(demo_tensor, save_path)
                self.save_parameters(save_path)
                raise e
            except KeyboardInterrupt as e:
                plt.plot(losses)
                plt.savefig(str(session_path.joinpath("loss_graph.png")))
                save_path = session_path.joinpath(f"{i}/")
                save_path.mkdir()
                self.inference(demo_tensor, save_path)
                self.save_parameters(save_path)
                raise e


            if i % save_steps == 0:
                save_path = session_path.joinpath(f"{i}/")
                save_path.mkdir()
                
                if i == 0:
                    grad_graph = torchviz.make_dot(loss, dict(self.q_theta.named_parameters()))
                    grad_graph.render(save_path.joinpath("grad_graph"))

                self.inference(demo_tensor, save_path)
                self.save_parameters(save_path)
                
            if i >= iterations: break
            # if i > 100 and past_10_variance < 1e-7: break
            
        plt.plot(losses)
        plt.savefig(str(session_path.joinpath("loss_graph.png")))
        # plt.show()

    def inference(self, input_tensor, save_dir=None):
        with torch.no_grad():
            h0, z0 = self.run_tau_network(input_tensor)
            z0_dists = torch.norm(z0, p=2, dim=1)
            clusterer = self.get_clusterer(h0, self.eps_coeff, self.num_clusters, clusterer_type="kmeans")
            cluster_ids, masks, m0, m1, m2 = self.generate_masks(h0, clusterer=clusterer)
            num_masks = len(cluster_ids)

            resized_masks = tfunc.resize(m0, size=input_tensor.shape[2:])
            masked_base_images = torch.concat([(input_tensor.squeeze()*mi).reshape((1,)+input_tensor.shape[1:]) for mi in resized_masks])
            print(masked_base_images.shape)

        if save_dir:
            draw_image(input_tensor, str(save_dir.joinpath("input_image.png")))
            draw_layers(h0, str(save_dir.joinpath("h0.png")))
            draw_layers(z0, str(save_dir.joinpath("z0.png")))
            draw_image(z0_dists, str(save_dir.joinpath("z0_dists.png")))
            draw_image(masks, str(save_dir.joinpath("masks.png")))
            draw_layers(m0, str(save_dir.joinpath("m0.png")))
            for i, mbi in enumerate(masked_base_images):
                draw_image(mbi, str(save_dir.joinpath(f"image_masked_{i}.png")))
        else:
            draw_image(input_tensor)
            draw_layers(h0)
            draw_layers(z0)
            draw_image(z0_dists)
            draw_image(masks)
            draw_layers(m0)

if __name__ == "__main__":
    for lr in [1e-3, 5e-4, 1e-4]:
        for clusterer in ["kmeans", "optics", "both"]:
            for loss_method in ["contrastive", "both"]:
                for num_clusters in range(4,16,2):
                    odin = ODIN(*make_dataloaders(), lr_tau=0.1, lr_theta=lr, lr_xi=0.1, clusters=4, fpn_dim=64)
                    odin.set_num_clusters(num_clusters)
                    try:
                        odin.train(iterations=2500, save_steps=250, clusterer_type=clusterer, loss_method=loss_method,
                                   session_id=f"{lr:.0e}-{clusterer}-{loss_method}-{num_clusters}")
                    except Exception as e:
                        print(f"Run {lr}-{clusterer}-{loss_method}-{num_clusters}: {e}")
                        continue