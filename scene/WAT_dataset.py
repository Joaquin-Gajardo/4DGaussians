'''
The following code is taken from the CLNeRF paper repository: https://github.com/IntelLabs/CLNeRF
'''

import os

import cv2
import imageio
from kornia import create_meshgrid
import numpy as np
from PIL import Image
import torch
from torchvision import transforms as T
from torch.utils.data import Dataset
from tqdm import tqdm

# WAT_colmap_utils.py is taken as-is from the CLNeRF paper repository.
# The logic on 4DGaussians.scene.colmap_loader.py is very similar but has some
# mofifications wrt to that here so we take the one from CLNeRF for the WAT dataset
from .WAT_colmap_utils import \
    read_cameras_binary, read_images_binary, read_points3d_binary 


###### CLNeRF.datasets.NGPA.ray_utils.py ######

def normalize(v):
    """Normalize a vector."""
    return v/np.linalg.norm(v)


def average_poses(poses, pts3d=None):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of 3d point cloud (if None, center of cameras).
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.
    
    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.
    Inputs:
        poses: (N_images, 3, 4)
        pts3d: (N, 3)

    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    if pts3d is not None:
        center = pts3d.mean(0)
    else:
        center = poses[..., 3].mean(0)

    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0)) # (3)

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0) # (3)

    # 4. Compute the x axis
    x = normalize(np.cross(y_, z)) # (3)

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(z, x) # (3)

    pose_avg = np.stack([x, y, z, center], 1) # (3, 4)

    return pose_avg


def center_poses(poses, pts3d=None):
    """
    See https://github.com/bmild/nerf/issues/34
    Inputs:
        poses: (N_images, 3, 4)
        pts3d: (N, 3) reconstructed point cloud

    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pts3d_centered: (N, 3) centered point cloud
    """

    pose_avg = average_poses(poses, pts3d) # (3, 4)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[:3] = pose_avg # convert to homogeneous coordinate for faster computation
                                 # by simply adding 0, 0, 0, 1 as the last row
    pose_avg_inv = np.linalg.inv(pose_avg_homo)
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1)) # (N_images, 1, 4)
    poses_homo = \
        np.concatenate([poses, last_row], 1) # (N_images, 4, 4) homogeneous coordinate

    poses_centered = pose_avg_inv @ poses_homo # (N_images, 4, 4)
    poses_centered = poses_centered[:, :3] # (N_images, 3, 4)

    if pts3d is not None:
        pts3d_centered = pts3d @ pose_avg_inv[:, :3].T + pose_avg_inv[:, 3:].T
        pts3d_centered = pts3d_centered[:, :3] # Dropping last column, probably an oversight
        
        # This below gives the same answer:
        # # Convert pts3d to homogeneous coordinates
        # pts3d_homo = np.concatenate([pts3d, np.ones((pts3d.shape[0], 1))], axis=1)
        # # Apply transformation
        # pts3d_centered = (pose_avg_inv @ pts3d_homo.T).T
        # # Convert back to 3D coordinates
        # pts3d_centered = pts3d_centered[:, :3]
        
        return poses_centered, pts3d_centered

    return poses_centered


@torch.cuda.amp.autocast(dtype=torch.float32)
def get_ray_directions(H, W, K, device='cpu', random=False, return_uv=False, flatten=True, crop_region = 'full'):
    """
    Get ray directions for all pixels in camera coordinate [right down front].
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        H, W: image height and width
        K: (3, 3) camera intrinsics
        random: whether the ray passes randomly inside the pixel
        return_uv: whether to return uv image coordinates

    Outputs: (shape depends on @flatten)
        directions: (H, W, 3) or (H*W, 3), the direction of the rays in camera coordinate
        uv: (H, W, 2) or (H*W, 2) image coordinates
    """
    grid = create_meshgrid(H, W, False, device=device)[0] # (H, W, 2)
    u, v = grid.unbind(-1)

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    if random:
        directions = \
            torch.stack([(u-cx+torch.rand_like(u))/fx,
                         (v-cy+torch.rand_like(v))/fy,
                         torch.ones_like(u)], -1)
    else: # pass by the center
        directions = \
            torch.stack([(u-cx+0.5)/fx, (v-cy+0.5)/fy, torch.ones_like(u)], -1)
    
    if crop_region  == 'left':
        directions = directions[:, :directions.shape[1]//2]

    elif crop_region == 'right':
        directions = directions[:, directions.shape[1]//2:]


    if flatten:
        directions = directions.reshape(-1, 3)
        grid = grid.reshape(-1, 2)

    if return_uv:
        return directions, grid
    return directions


###### CLNeRF.datasets.base.py ######

# class BaseDataset(Dataset):
#     """
#     Define length and sampling method
#     """
#     def __init__(self, root_dir, split='train', downsample=1.0):
#         self.root_dir = root_dir
#         self.split = split
#         self.downsample = downsample

#     def read_intrinsics(self):
#         raise NotImplementedError

#     def __len__(self):
#         if self.split.startswith('train'):
#             return 1000
#         return len(self.poses)

#     def __getitem__(self, idx):
#         if self.split.startswith('train'):
#             # training pose is retrieved in train.py
#             if self.ray_sampling_strategy == 'all_images': # randomly select images
#                 img_idxs = np.random.choice(len(self.poses), self.batch_size)
#             elif self.ray_sampling_strategy == 'same_image': # randomly select ONE image
#                 img_idxs = np.random.choice(len(self.poses), 1)[0]
#             # randomly select pixels
#             pix_idxs = np.random.choice(self.img_wh[0]*self.img_wh[1], self.batch_size)
#             rays = self.rays[img_idxs, pix_idxs]
#             sample = {'img_idxs': img_idxs, 'pix_idxs': pix_idxs,
#                       'rgb': rays[:, :3]}
#             if self.rays.shape[-1] == 4: # HDR-NeRF data
#                 sample['exposure'] = rays[:, 3:]
#         else:
#             sample = {'pose': self.poses[idx], 'img_idxs': idx}
#             if len(self.rays)>0: # if ground truth available
#                 rays = self.rays[idx]
#                 sample['rgb'] = rays[:, :3]
#                 if rays.shape[1] == 4: # HDR-NeRF data
#                     sample['exposure'] = rays[0, 3] # same exposure for all rays

#         return sample

###### CLNeRF.datasets.NGPA.colmap.py ######

def name_to_task(img_paths):
    tasks, task_list, test_ids = [], [], []
    for img_path in img_paths:
        task_folder_name = os.path.basename(os.path.dirname(img_path))
        if task_folder_name not in tasks:
            tasks.append(task_folder_name)
        task_list.append(tasks.index(task_folder_name))

    img_count = 0
    task_curr = -1
    for i in range(len(task_list)):
        if task_list[i] > task_curr:
            task_curr, img_count = task_list[i], 0
            test_ids.append(i)
        elif img_count % 8 == 0:
            test_ids.append(i)
        img_count += 1
    return task_list, test_ids
    

class ColmapDataset_NGPA(Dataset):

    def __init__(self, root_dir, split='train', downsample=1.0, **kwargs):
        self.root_dir = root_dir
        self.split = split
        self.downsample = downsample
        self.transform = T.ToTensor()

        self.read_intrinsics()

        if kwargs.get('read_meta', True):
            self.read_meta(split, **kwargs)

    def read_intrinsics(self):
        # Step 1: read and scale intrinsics (same for all images)
        camdata = read_cameras_binary(
            os.path.join(self.root_dir, 'sparse/0/cameras.bin'))
        h = int(camdata[1].height * self.downsample)
        w = int(camdata[1].width * self.downsample)
        self.img_wh = (w, h)
        print("self.img_wh = {}".format(self.img_wh))

        if camdata[1].model == 'SIMPLE_RADIAL':
            fx = fy = camdata[1].params[0] * self.downsample
            cx = camdata[1].params[1] * self.downsample
            cy = camdata[1].params[2] * self.downsample
        elif camdata[1].model in ['PINHOLE', 'OPENCV']:
            fx = camdata[1].params[0] * self.downsample
            fy = camdata[1].params[1] * self.downsample
            cx = camdata[1].params[2] * self.downsample
            cy = camdata[1].params[3] * self.downsample
        else:
            raise ValueError(
                f"Please parse the intrinsics for camera model {camdata[1].model}!"
            )
        self.K = torch.FloatTensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        self.directions = get_ray_directions(h, w, self.K)

    def read_meta(self, split, **kwargs):
        # Step 2: correct poses
        # read extrinsics (of successfully reconstructed images)
        imdata = read_images_binary(
            os.path.join(self.root_dir, 'sparse/0/images.bin'))
        img_names = [imdata[k].name for k in imdata]
        perm = np.argsort(img_names)
        if '360_v2' in self.root_dir and self.downsample < 1:  # mipnerf360 data
            folder = f'images_{int(1/self.downsample)}'
        else:
            folder = 'images'
        # read successfully reconstructed images and ignore others
        img_paths = [
            os.path.join(self.root_dir, folder, name)
            for name in sorted(img_names)
        ]
        # get the task id
        task_ids, test_img_ids = name_to_task(img_paths)

        w2c_mats = []
        bottom = np.array([[0, 0, 0, 1.]])
        for k in imdata:
            im = imdata[k]
            R = im.qvec2rotmat()
            t = im.tvec.reshape(3, 1)
            w2c_mats += [
                np.concatenate([np.concatenate([R, t], 1), bottom], 0)
            ]
        w2c_mats = np.stack(w2c_mats, 0)
        poses = np.linalg.inv(w2c_mats)[
            perm, :3]  # (N_images, 3, 4) cam2world matrices

        pts3d = read_points3d_binary(
            os.path.join(self.root_dir, 'sparse/0/points3D.bin'))
        self.pts3d_rgb = np.array([pts3d[k].rgb for k in pts3d])  # (N, 3)
        pts3d = np.array([pts3d[k].xyz for k in pts3d])  # (N, 3)

        self.poses, self.pts3d = center_poses(poses, pts3d)

        # compute near far
        self.xyz_world = pts3d
        xyz_world_h = np.concatenate(
            [self.xyz_world, np.ones((len(self.xyz_world), 1))], -1)
        # Compute near and far bounds for each image individually
        self.nears, self.fars = {}, {}  # {id_: distance}
        for i, id_ in enumerate(imdata):
            xyz_cam_i = (xyz_world_h @ w2c_mats[i].T
                         )[:, :3]  # xyz in the ith cam coordinate
            xyz_cam_i = xyz_cam_i[
                xyz_cam_i[:,
                          2] > 0]  # filter out points that lie behind the cam
            self.nears[id_] = np.percentile(xyz_cam_i[:, 2], 0.1)
            self.fars[id_] = np.percentile(xyz_cam_i[:, 2], 99.9)

        max_far = np.fromiter(self.fars.values(), np.float32).max()
        min_near = np.fromiter(self.nears.values(), np.float32).min()

        scale = max_far / 8.0
        print("[test] near_far = {}/{}, scale = {}".format(
            min_near, max_far, scale))
        self.poses[..., 3] /= scale
        self.pts3d /= scale

        self.rays = []
        self.ts = []

        # train test split
        if split == 'train':
            self.img_paths = [
                x for i, x in enumerate(img_paths) if i not in test_img_ids
            ]
            self.poses = np.array(
                [x for i, x in enumerate(self.poses) if i not in test_img_ids])
            task_ids = [
                x for i, x in enumerate(task_ids) if i not in test_img_ids
            ]
        elif split == 'test':
            self.img_paths = [
                x for i, x in enumerate(img_paths) if i in test_img_ids
            ]
            self.poses = np.array(
                [x for i, x in enumerate(self.poses) if i in test_img_ids])
            task_ids = [x for i, x in enumerate(task_ids) if i in test_img_ids]

        # print(f'Loading {len(img_paths)} {split} images ...')
        # for i, img_path in enumerate(tqdm(self.img_paths)):
        #     buf = []  # buffer for ray attributes: rgb, etc

        #     img = read_image(img_path, self.img_wh, blend_a=False)
        #     img = torch.FloatTensor(img)
        #     #buf += [img]

        #     #self.rays += [torch.cat(buf, 1)]
        #     self.ts += [task_ids[i]]

        print(f'Preparing {split} split: {len(self.img_paths)} images ...')
        # self.rays = torch.stack(self.rays)  # (N_images, hw, ?)
        self.poses = torch.FloatTensor(self.poses)  # (N_images, 3, 4)
        self.ts = torch.tensor(task_ids).int()  # (N_images)

    def __getitem__(self, idx):
        img = self.read_image(self.img_paths[idx])
        #img = torch.tensor(np.array(img)) # (H, W, 3), dtype=torch.uint8
        #img = img.resize((int(self.img_wh[0]/2), int(self.img_wh[1]/2)))
        img = self.transform(img)
        return img, self.poses[idx], self.ts[idx]
    
    def __len__(self):
        return len(self.img_paths)

    def read_image(self, img_path):
        img = Image.open(img_path)
        img = img.resize(self.img_wh, Image.LANCZOS)
        return img
    
    def read_image2(img_path, img_wh):
        img = imageio.imread(img_path)
        img = cv2.resize(img, img_wh)
        return img
    
# def format_WAT_infos(dataset, split):
#     cameras = []
#     image = dataset[0][0]
    
#     if split == "train":
#         for idx in tqdm(range(len(dataset))):
#             image_path = None
#             image_name = f"{idx}"
#             time = dataset.ts[idx].item()  # Assuming ts is the time information
            
#             # Get pose information
#             pose = dataset.poses[idx]
#             R = pose[:3, :3].T  # Transpose to get from world to camera
#             T = -R @ pose[:3, 3]  # Convert position to translation
            
#             FovX = focal2fov(dataset.K[0, 0].item(), image.shape[1])
#             FovY = focal2fov(dataset.K[1, 1].item(), image.shape[0])
            
#             cameras.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
#                                       image_path=image_path, image_name=image_name, 
#                                       width=image.shape[1], height=image.shape[0],
#                                       time=time, mask=None))

#     return cameras