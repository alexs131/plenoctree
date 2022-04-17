import torch
import os
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import imageio
from argparse import ArgumentParser
from annoy import AnnoyIndex

import nerfvis

dir_path = os.path.dirname(os.path.realpath(__file__))

def get_opts():
    parser = ArgumentParser()
    parser.add_argument('--root_dir', type=str,
                        default=os.path.join(dir_path, 'silica'),
                        help='root directory of dataset')
    parser.add_argument('--dataset_name', type=str, default='llff',
                        choices=['blender', 'llff'],
                        help='which dataset to validate')
    parser.add_argument('--scene_name', type=str, default='lego',
                        help='scene name, used as output folder name')
    parser.add_argument('--split', type=str, default='test',
                        help='test or test_train')
    parser.add_argument('--img_wh', nargs="+", type=int, default=[504, 378],
                        help='resolution (img_w, img_h) of the image')
    parser.add_argument('--spheric_poses', default=True, action="store_true",
                        help='whether images are taken in spheric poses (for llff)')

    parser.add_argument('--chunk', type=int, default=32*1024*4,
                        help='chunk size to split the input to avoid OOM')

    parser.add_argument('--ckpt_path', type=str, default=os.path.join(dir_path, 'silica/silica.ckpt'),
                        help='pretrained checkpoint path to load')
    parser.add_argument('--port', type=int, default=8889,
                        help='port to run server')

    return parser.parse_args()


if __name__ == "__main__":
    args = get_opts()
    w, h = args.img_wh

    scene = nerfvis.Scene(args.scene_name)

    idx = AnnoyIndex(3, metric="euclidean")
    idx.load("lego.ann")
    rgba_data = torch.from_numpy(np.load("legodata.npz")["rgba_data"]).cuda()

    @torch.no_grad()
    def nerf_func(points):
        # points [B, 3]
        rgba = []
        for i in range(points.shape[0]):
            id = idx.get_nns_by_vector(points[i], 1)
            rgba.append(rgba_data[id])
        result = torch.stack(rgba)
        return result[..., :-1], result[..., -1:]


    # This will project to SH. You can change sh_deg to use higher-degree SH 
    # or increase sh_proj_sample_count to improve the accuracy of the projection
    scene.add_camera_frustum(name="cam1",image_width=w, image_height=h, r=np.eye(3), t=np.array([0.0,0.0,2.0]), update_view=True)
    scene.set_nerf(nerf_func, center=[0.0, 0.0, 0.0], radius=1.5, use_dirs=False, sigma_thresh=0.005)
    scene.nerf.save("legooct.npz")
    scene.display(port=args.port)
