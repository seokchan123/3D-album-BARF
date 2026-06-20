import numpy as np
import os,sys,time
import torch
import torch.nn.functional as torch_F
import torchvision
import torchvision.transforms.functional as torchvision_F
import PIL
import imageio
from easydict import EasyDict as edict
import json
import pickle

from . import base
import camera
from util import log,debug

class Dataset(base.Dataset):

    def __init__(self,opt,split="train",subset=None):
        self.raw_H,self.raw_W = 3000,4000
        super().__init__(opt,split)
        self.root = opt.data.root or "data/llff"
        self.path = "{}/{}".format(self.root,opt.data.scene)
        self.path_image = "{}/images".format(self.path)

        self.path_depth = "{}/depth".format(self.path)

        image_fnames = sorted(os.listdir(self.path_image))
        poses_raw,bounds = self.parse_cameras_and_bounds(opt)
        self.list = list(zip(image_fnames,poses_raw,bounds))
        # manually split train/val subsets
        num_val_split = int(len(self)*opt.data.val_ratio)
        self.list = self.list[:-num_val_split] if split=="train" else self.list[-num_val_split:]
        if subset: self.list = self.list[:subset]
        # preload dataset
        if opt.data.preload:
            self.images = self.preload_threading(opt,self.get_image)
            self.cameras = self.preload_threading(opt,self.get_camera,data_str="cameras")

    def prefetch_all_data(self,opt):
        assert(not opt.data.augment)
        # pre-iterate through all samples and group together
        self.all = torch.utils.data._utils.collate.default_collate([s for s in self])

    def parse_cameras_and_bounds(self,opt):
        fname = "{}/poses_bounds.npy".format(self.path)
        data = torch.tensor(np.load(fname),dtype=torch.float32)
        # parse cameras (intrinsics and poses)
        cam_data = data[:,:-2].view([-1,3,5]) # [N,3,5]
        poses_raw = cam_data[...,:4] # [N,3,4]
        poses_raw[...,0],poses_raw[...,1] = poses_raw[...,1],-poses_raw[...,0]
        raw_H,raw_W,self.focal = cam_data[0,:,-1]
        assert(self.raw_H==raw_H and self.raw_W==raw_W)
        # parse depth bounds
        bounds = data[:,-2:] # [N,2]
        scale = 1./(bounds.min()*0.75) # not sure how this was determined
        poses_raw[...,3] *= scale
        bounds *= scale
        # roughly center camera poses
        poses_raw = self.center_camera_poses(opt,poses_raw)
        return poses_raw,bounds

    def center_camera_poses(self,opt,poses):
        # compute average pose
        center = poses[...,3].mean(dim=0)
        v1 = torch_F.normalize(poses[...,1].mean(dim=0),dim=0)
        v2 = torch_F.normalize(poses[...,2].mean(dim=0),dim=0)
        v0 = v1.cross(v2)
        pose_avg = torch.stack([v0,v1,v2,center],dim=-1)[None] # [1,3,4]
        # apply inverse of averaged pose
        poses = camera.pose.compose([poses,camera.pose.invert(pose_avg)])
        return poses

    def get_all_camera_poses(self,opt):
        pose_raw_all = [tup[1] for tup in self.list]
        pose_all = torch.stack([self.parse_raw_camera(opt,p) for p in pose_raw_all],dim=0)
        return pose_all

    def __getitem__(self,idx):
        opt = self.opt
        sample = dict(idx=idx)
        aug = self.generate_augmentation(opt) if self.augment else None
        
        # ==========================================
        # get_image에서 반환하는 값 2개를 모두 받도록 수정
        # ==========================================
        if opt.data.preload:
            # 만약 원본 코드가 preload를 쓰고 있었다면 데이터 구조에 맞춰 꺼냅니다.
            image = self.images[idx][0]
            depth_map = self.images[idx][1]
        else:
            image, depth_map = self.get_image(opt,idx)

        image = self.preprocess_image(opt,image,aug=aug)
        intr,pose = self.cameras[idx] if opt.data.preload else self.get_camera(opt,idx)
        intr,pose = self.preprocess_camera(opt,intr,pose,aug=aug)
        
        # 뎁스 맵도 이미지와 똑같이 사이즈를 줄여주거나 전처리(1채널 이미지 취급)
        # resize 및 텐서 변환 (opt.H, opt.W 크기에 맞춤)
        depth_map = depth_map.unsqueeze(0).unsqueeze(0) # [1, 1, H, W]
        depth_map = torch_F.interpolate(depth_map, size=(opt.H, opt.W), mode='bilinear', align_corners=False)
        depth_map = depth_map.squeeze(0) # [1, H, W]

        sample.update(
            image=image,
            depth_gt=depth_map, # <--- 쟁반에 깊이 데이터 추가!
            intr=intr,
            pose=pose,
        )
        return sample

    def get_image(self,opt,idx):
        image_fname = "{}/{}".format(self.path_image,self.list[idx][0])
        image = PIL.Image.fromarray(imageio.imread(image_fname)) 

        # ==========================================
        # 해당 이미지와 짝꿍인 Depth 데이터(.npy) 읽기
        # ==========================================
        # ex) image_001.jpg -> image_001.npy 로 이름 변환
        base_name = os.path.splitext(self.list[idx][0])[0]
        depth_fname = "{}/{}.npy".format(self.path_depth, base_name)
        
        try:
            depth_data = np.load(depth_fname)
            # 데이터를 파이토치 텐서로 변환
            depth_tensor = torch.from_numpy(depth_data).float()
        except:
            # 혹시 깊이 맵 파일이 없으면 에러 방지용으로 0으로 채운 텐서 반환
            depth_tensor = torch.zeros((self.raw_H, self.raw_W)).float()
            
        # image만 반환하던 것을 depth_tensor와 함께 반환하도록 수정
        return image, depth_tensor

    def get_camera(self,opt,idx):
        intr = torch.tensor([[self.focal,0,self.raw_W/2],
                             [0,self.focal,self.raw_H/2],
                             [0,0,1]]).float()
        pose_raw = self.list[idx][1]
        pose = self.parse_raw_camera(opt,pose_raw)
        return intr,pose

    def parse_raw_camera(self,opt,pose_raw):
        pose_flip = camera.pose(R=torch.diag(torch.tensor([1,-1,-1])))
        pose = camera.pose.compose([pose_flip,pose_raw[:3]])
        pose = camera.pose.invert(pose)
        pose = camera.pose.compose([pose_flip,pose])
        return pose
