from torch.utils.data import Dataset
from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal, focal2fov
import torch
from utils.camera_utils import loadCam
from utils.graphics_utils import focal2fov
class FourDGSdataset(Dataset):
    def __init__(
        self,
        dataset,
        args,
        dataset_type
    ):
        self.dataset = dataset
        self.args = args
        self.dataset_type=dataset_type
    def __getitem__(self, index):
        # breakpoint()

        if self.dataset_type != "PanopticSports":
            try:
                image, pose, time = self.dataset[index]
                if self.dataset_type == "WAT":
                    R = pose[:3, :3].numpy()
                    T = pose[:3, 3].numpy()
                    FovX = focal2fov(self.dataset.K[0, 0].item(), self.dataset.img_wh[0])
                    FovY = focal2fov(self.dataset.K[1, 1].item(), self.dataset.img_wh[1])
                else:
                    R,T = pose
                    FovX = focal2fov(self.dataset.focal[0], image.shape[2])
                    FovY = focal2fov(self.dataset.focal[0], image.shape[1])
                mask = None
            except:
                caminfo = self.dataset[index]
                image = caminfo.image
                R = caminfo.R
                T = caminfo.T
                FovX = caminfo.FovX
                FovY = caminfo.FovY
                time = caminfo.time
    
                mask = caminfo.mask
            camera = Camera(colmap_id=index,R=R,T=T,FoVx=FovX,FoVy=FovY,image=image,gt_alpha_mask=None,
                            image_name=f"{index}",uid=index,data_device=torch.device("cuda"), time=time,
                            mask=mask, scale=self.args.scene_scale)
            return camera
        else:
            return self.dataset[index]
    def __len__(self):
        
        return len(self.dataset)
