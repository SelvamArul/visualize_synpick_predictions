import torch
import torch.nn.functional as F
import stillleben as sl
import argparse
import json
from pathlib import Path
import visdom
from PIL import Image
import numpy as np
vis = visdom.Visdom(env="synpick_vis", port=8097)
vis.close(env="synpick_vis")

sl.init()


obj_list = ['002_master_chef_can',
            '003_cracker_box',
            '004_sugar_box',
            '005_tomato_soup_can',
            '006_mustard_bottle',
            '007_tuna_fish_can',
            '008_pudding_box',
            '009_gelatin_box',
            '010_potted_meat_can',
            '011_banana',
            '019_pitcher_base',
            '021_bleach_cleanser',
            '024_bowl',
            '025_mug',
            '035_power_drill',
            '036_wood_block',
            '037_scissors',
            '040_large_marker',
            '051_large_clamp',
            '052_extra_large_clamp',
            '061_foam_brick',
]

def label_2_mesh(label):
    id = label.split('_')[1]
    id = int(id) -1
    return obj_list[id]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test PoseRBPF on YCB Video or T-LESS Datasets (RGB)')
    parser.add_argument('--results_path', dest='results_path', help='results.pth.tar',
                        default='/home/user/periyasa/workspace/cosypose/local_data/results/synpick--137577/dataset=synpick/results.pth.tar', type=str)
    parser.add_argument('--models_path', dest='models_path', help='ycb models',
                        default='/home/cache/ycbv/models_bop-compat', type=str)
    parser.add_argument('--dataset_path', dest='dataset_path', help='synpick dataset',
                        default="/home/cache/synpick/test_pick3", type=str)
    parser.add_argument('--save_img', action='store_true')
    args = parser.parse_args()
    results = torch.load(args.results_path)['predictions']
    poses = results ['maskrcnn_detections/refiner/iteration=4'].poses
    bboxes = results ['maskrcnn_detections/detections'].bboxes
    infos = results ['maskrcnn_detections/detections'].infos


    dataset_path = Path(args.dataset_path)
    models_path = Path(args.models_path)
    for index, row in infos.iterrows():
        label = row['label']
        scene_id = row['scene_id']
        view_id = row['view_id']
        rgb_path = dataset_path / f'{scene_id:06d}' / 'rgb' / f'{view_id:06d}.jpg'
        cam_path =   dataset_path / f'{scene_id:06d}' / 'scene_camera.json'
        print ("cam_path ", cam_path)
        with open(str(cam_path)) as jf:
            scene_camera = json.load(jf)
        scene_camera  = scene_camera[str(scene_id)]

        print ("scene_camera ", scene_camera)
        cam_K = scene_camera['cam_K']
        fx, cx, fy, cy = cam_K[0], cam_K[2], cam_K[4], cam_K[5]
        print ("fx, cx, fy, cy ", fx, cx, fy, cy)
        
        scene = sl.Scene((1920, 1080))
        scene.ambient_light = torch.tensor([0.9, 0.9, 0.9])
        scene.background_color = torch.tensor([0.1, 0.1, 0.1, 1.0])
        eye = torch.eye(4)
        scene.set_camera_pose(eye)
        scene.choose_random_light_position()
        scene.set_camera_intrinsics (fx, fy, cx, cy)
        mesh = sl.Mesh(models_path / f'{label}.ply')
        pt = torch.eye(4)
        pt[:3,:3] *= 0.001
        mesh.pretransform = pt
        obj = sl.Object(mesh)
        scene.add_object(obj)
        pose = poses[index]
        print ('pose \n', pose)
        obj.set_pose(pose)
        renderer = sl.RenderPass()
        result = renderer.render(scene)
        result.rgb()
        vis.close(env="synpick_vis")

        import ipdb; ipdb.set_trace()
        pred = result.rgb()[:,:,:3].cpu().permute(2,0,1) / 255.
        pred = F.interpolate(pred.unsqueeze(0), scale_factor=0.5, mode='bilinear')
        vis.images(pred.numpy())
        rgb_scene = np.array(Image.open(rgb_path))
        gt = torch.from_numpy(rgb_scene.transpose(2,0,1)) / 255.
        gt = F.interpolate(gt.unsqueeze(0), scale_factor=0.5, mode='bilinear')
        vis.images(gt.numpy())
        
        if args.save_img:
            Image.fromarray(result.rgb()[:,:,:3].cpu().numpy()).save('pred.jpeg')
            rgb_scene.save("scene.jpeg")
        yes = input("press <enter> to continue")