import torch
import torchvision.transforms as tvf

from ..utils.base_model import BaseModel


class OpenIBL(BaseModel):
    default_conf = {
        'model_name': 'vgg16_netvlad',
    }
    required_inputs = ['image']

    def _init(self, conf):
        try:
            # first try load from cache directory
            import os
            cache_dir = os.path.expanduser('~/.cache/torch/hub/yxgeee_OpenIBL_master')
            weight_path = os.path.expanduser('~/.cache/torch/hub/checkpoints/vgg16_netvlad.pth')
            self.net = torch.hub.load( 
                 cache_dir, conf['model_name'], source='local')
            self.net.load_state_dict(torch.load(
                weight_path, map_location=next(self.net.parameters()).device))
            self.net.eval()
        except:
            self.net = torch.hub.load(
                'yxgeee/OpenIBL', conf['model_name'], pretrained=True).eval()
        mean = [0.48501960784313836, 0.4579568627450961, 0.4076039215686255]
        std = [0.00392156862745098, 0.00392156862745098, 0.00392156862745098]
        self.norm_rgb = tvf.Normalize(mean=mean, std=std)

    def _forward(self, data):
        image = self.norm_rgb(data['image'])
        desc = self.net(image)
        return {
            'global_descriptor': desc,
        }
