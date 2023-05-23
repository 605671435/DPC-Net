# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import math
import os
import shutil

import torch
import os.path as osp
import pkg_resources
import re
from pathlib import Path

import mmcv
import numpy as np
from mmcv.transforms import Compose
from mmengine.config import Config, DictAction
from mmengine.utils import mkdir_or_exist
from torch.nn import BatchNorm1d, BatchNorm2d, GroupNorm, LayerNorm

from seg.registry import VISUALIZERS
from mmseg.apis import init_model, inference_model
from seg.utils import register_all_modules
from mmseg.models.utils import resize
from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget

try:
    import pytorch_grad_cam
    from pytorch_grad_cam import (EigenCAM, EigenGradCAM, GradCAM,
                                  GradCAMPlusPlus, LayerCAM, XGradCAM)
    from pytorch_grad_cam.activations_and_gradients import \
        ActivationsAndGradients
    from pytorch_grad_cam.utils.image import show_cam_on_image
except ImportError:
    raise ImportError('Please run `pip install "grad-cam>=1.3.6"` to install '
                      '3rd party package pytorch_grad_cam.')

# Supported grad-cam type map
METHOD_MAP = {
    'gradcam': GradCAM,
    'gradcam++': GradCAMPlusPlus,
    'xgradcam': XGradCAM,
    'eigencam': EigenCAM,
    'eigengradcam': EigenGradCAM,
    'layercam': LayerCAM,
}


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize CAM')
    parser.add_argument('img', help='Image file')
    parser.add_argument('seg', help='Segmentation file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--target-layers',
        default=[],
        nargs='+',
        type=str,
        help='The target layers to get CAM, if not set, the tool will '
             'specify the norm layer in the last block. Backbones '
             'implemented by users are recommended to manually specify'
             ' target layers in commmad statement.')
    parser.add_argument(
        '--preview-model',
        default=False,
        action='store_true',
        help='To preview all the model layers')
    parser.add_argument(
        '--method',
        default='gradcam++',
        help='Type of method to use, supports '
             f'{", ".join(list(METHOD_MAP.keys()))}.')
    parser.add_argument(
        '--target-category',
        default=[],
        nargs='+',
        type=int,
        help='The target category to get CAM, default to use result '
             'get from given model.')
    parser.add_argument(
        '--eigen-smooth',
        default=True,
        action='store_true',
        help='Reduce noise by taking the first principle componenet of '
             '``cam_weights*activations``')
    parser.add_argument(
        '--aug-smooth',
        default=False,
        action='store_true',
        help='Wether to use test time augmentation, default not to use')
    parser.add_argument(
        '--save-path',
        type=Path,
        help='The path to save visualize cam image, default not to save.')
    parser.add_argument('--device', default='cpu', help='Device to use cpu')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')
    args = parser.parse_args()
    if args.method.lower() not in METHOD_MAP.keys():
        raise ValueError(f'invalid CAM type {args.method},'
                         f' supports {", ".join(list(METHOD_MAP.keys()))}.')

    return args


class ActivationsWrapper:

    def __init__(self, model, target_layers):
        self.model = model
        self.activations = []
        self.handles = []
        self.image = None
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(self.save_activation))

    def save_activation(self, module, input, output):
        self.activations.append(output)

    def __call__(self, img_path):
        self.activations = []
        results = inference_model(self.model, img_path)
        return results, self.activations

    def release(self):
        for handle in self.handles:
            handle.remove()


def build_reshape_transform(model, args):
    """Build reshape_transform for `cam.activations_and_grads`, which is
    necessary for ViT-like networks."""

    # ViT_based_Transformers have an additional clstoken in features
    def check_shape(tensor):
        if isinstance(tensor, tuple):
            assert len(tensor[0].size()) != 3, \
                (f"The input feature's shape is {tensor.size()}, and it seems "
                 'to have been flattened or from a vit-like network. '
                 "Please use `--vit-like` if it's from a vit-like network.")
            return tensor
        assert len(tensor.size()) != 3, \
            (f"The input feature's shape is {tensor.size()}, and it seems "
             'to have been flattened or from a vit-like network. '
             "Please use `--vit-like` if it's from a vit-like network.")
        return tensor

    return check_shape


def init_cam(method, model, target_layers, use_cuda, reshape_transform):
    """Construct the CAM object once, In order to be compatible with mmcls,
    here we modify the ActivationsAndGradients object."""
    GradCAM_Class = METHOD_MAP[method.lower()]
    cam = GradCAM_Class(
        model=model, target_layers=target_layers, use_cuda=use_cuda)
    # Release the original hooks in ActivationsAndGradients to use
    # ActivationsAndGradients.
    cam.activations_and_grads.release()
    cam.activations_and_grads = ActivationsAndGradients(
        cam.model, cam.target_layers, reshape_transform)

    return cam


def get_layer(layer_str, model):
    """get model layer from given str."""
    cur_layer = model
    layer_names = layer_str.strip().split('.')

    def get_children_by_name(model, name):
        try:
            return getattr(model, name)
        except AttributeError as e:
            raise AttributeError(
                e.args[0] +
                '. Please use `--preview-model` to check keys at first.')

    def get_children_by_eval(model, name):
        try:
            return eval(f'model{name}', {}, {'model': model})
        except (AttributeError, IndexError) as e:
            raise AttributeError(
                e.args[0] +
                '. Please use `--preview-model` to check keys at first.')

    for layer_name in layer_names:
        match_res = re.match('(?P<name>.+?)(?P<indices>(\\[.+\\])+)',
                             layer_name)
        if match_res:
            layer_name = match_res.groupdict()['name']
            indices = match_res.groupdict()['indices']
            cur_layer = get_children_by_name(cur_layer, layer_name)
            cur_layer = get_children_by_eval(cur_layer, indices)
        else:
            cur_layer = get_children_by_name(cur_layer, layer_name)

    return cur_layer


def show_cam_grad(grayscale_cam, src_img, title, out_path=None, return_img=False):
    """fuse src_img and grayscale_cam and show or save."""
    grayscale_cam = grayscale_cam[0, :]
    src_img = np.float32(src_img) / 255
    visualization_img = show_cam_on_image(
        src_img, grayscale_cam, use_rgb=True, image_weight=0.5)
    if return_img:
        return visualization_img
    if out_path:
        mmcv.imwrite(visualization_img, str(out_path))
        print('cam saved to {}'.format(str(out_path)))
    else:
        mmcv.imshow(visualization_img, win_name=title)


def get_default_traget_layers(model, args):
    """get default target layers from given model, here choose nrom type layer
    as default target layer."""
    norm_layers = []
    for m in model.backbone.modules():
        if isinstance(m, (BatchNorm2d, LayerNorm, GroupNorm, BatchNorm1d)):
            norm_layers.append(m)
    if len(norm_layers) == 0:
        raise ValueError(
            '`--target-layers` is empty. Please use `--preview-model`'
            ' to check keys at first and then specify `target-layers`.')
    # if the model is CNN model or Swin model, just use the last norm
    # layer as the target-layer, if the model is ViT model, the final
    # classification is done on the class token computed in the last
    # attention block, the output will not be affected by the 14x14
    # channels in the last layer. The gradient of the output with
    # respect to them, will be 0! here use the last 3rd norm layer.
    # means the first norm of the last decoder block.
    if args.vit_like:
        if args.num_extra_tokens:
            num_extra_tokens = args.num_extra_tokens
        elif hasattr(model.backbone, 'num_extra_tokens'):
            num_extra_tokens = model.backbone.num_extra_tokens
        else:
            raise AttributeError('Please set num_extra_tokens in backbone'
                                 " or using 'num-extra-tokens'")

        # if a vit-like backbone's num_extra_tokens bigger than 0, view it
        # as a VisionTransformer backbone, eg. DeiT, T2T-ViT.
        if num_extra_tokens >= 1:
            print('Automatically choose the last norm layer before the '
                  'final attention block as target_layer..')
            return [norm_layers[-3]]
    print('Automatically choose the last norm layer as target_layer.')
    target_layers = [norm_layers[-1]]
    return target_layers


class SegmentationModelOutputWrapper(torch.nn.Module):
    def __init__(self, model, ori_size):
        super(SegmentationModelOutputWrapper, self).__init__()
        self.model = model
        self.ori_size = ori_size

    def forward(self, x):
        seg_logits = self.model(x)
        seg_logits = resize(
            seg_logits,
            size=self.ori_size,
            mode='bilinear')
        return seg_logits

def label_mapping(label):
    """Label mapping from TransUNet paper setting. It only has 9 classes, which
    are 'background', 'aorta', 'gallbladder', 'left_kidney', 'right_kidney',
    'liver', 'pancreas', 'spleen', 'stomach', respectively. Other foreground
    classes in original dataset are all set to background.

    More details could be found here: https://arxiv.org/abs/2102.04306
    """
    maped_label = np.zeros_like(label)
    maped_label[label == 8] = 1
    maped_label[label == 4] = 2
    maped_label[label == 3] = 3
    maped_label[label == 2] = 4
    maped_label[label == 6] = 5
    maped_label[label == 11] = 6
    maped_label[label == 1] = 7
    maped_label[label == 7] = 8
    return maped_label

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    register_all_modules()
    # build the model from a config file and a checkpoint file
    ori_model = init_model(cfg, args.checkpoint, device=args.device)

    visualizer = VISUALIZERS.build(cfg.visualizer)

    classes = ori_model.dataset_meta['classes']
    if args.preview_model:
        print(ori_model)
        print('\n Please remove `--preview-model` to get the CAM.')
        return

    # apply transform and perpare data
    transforms = Compose(cfg.val_dataloader.dataset.pipeline)
    transforms_src = copy.deepcopy(transforms)
    transforms_src.transforms.pop(1)

    data = transforms({'img_path': args.img,
                       'seg_map_path': args.seg,
                       'reduce_zero_label': None,
                       'seg_fields': []})
    data_src = transforms_src({'img_path': args.img,
                               'seg_map_path': args.seg,
                               'reduce_zero_label': None,
                               'seg_fields': []})
    src_img = copy.deepcopy(data_src['inputs']).numpy().transpose(1, 2, 0)
    data['inputs'] = data['inputs'].unsqueeze(0)
    data['data_samples'] = [data['data_samples']]
    data = ori_model.data_preprocessor(data, False)

    predict = ori_model.predict(data['inputs'])
    pred_mask = np.float32(predict[0].pred_sem_seg.data.cpu().numpy())[0]
    gt_mask = np.float32(data['data_samples'][0].gt_sem_seg.data.cpu().numpy())[0]
    # gt_mask = label_mapping(gt_mask)

    # model = Encoder_Decoder(backbone=model.backbone, decode_head=model.decode_head, ori_size=data['inputs'].shape[-2:])
    model = SegmentationModelOutputWrapper(ori_model, ori_size=data['inputs'].shape[-2:])
    model.eval()
    # output = model(data['inputs'])
    # build target layers
    if args.target_layers:
        target_layers = [
            get_layer(layer, model) for layer in args.target_layers
        ]
    else:
        target_layers = get_default_traget_layers(model, args)

    # activations_wrapper = ActivationsWrapper(model, target_layers)

    # init a cam grad calculator
    use_cuda = ('cuda' in args.device)
    reshape_transform = build_reshape_transform(model, args)
    cam = init_cam(args.method, model, target_layers, use_cuda,
                   reshape_transform)
    gt_cls = np.unique(gt_mask).astype(np.uint8)
    num_classes = len(gt_cls)

    save_path = osp.join(args.save_path,
                         osp.splitext(osp.basename(args.config))[0],
                         osp.splitext(osp.basename(args.img))[0] + f'_{num_classes}_classes')
    mkdir_or_exist(save_path)
    shutil.copy(args.img, osp.join(save_path, 'ori_img.jpg'))

    # vis_gt = visualizer.draw_sem_seg(data_src['inputs'].cpu().numpy().transpose(1, 2, 0),
    #                                  data_src['data_samples'].gt_sem_seg,
    #                                  ori_model.dataset_meta['classes'],
    #                                  ori_model.dataset_meta['palette'])
    #
    # vis_pred = visualizer.draw_sem_seg(data_src['inputs'].cpu().numpy().transpose(1, 2, 0),
    #                                    predict[0].pred_sem_seg,
    #                                    ori_model.dataset_meta['classes'],
    #                                    ori_model.dataset_meta['palette'])
    #
    # mmcv.imwrite(vis_gt, osp.join(save_path, 'gt_whole.png'))
    # mmcv.imwrite(vis_pred, osp.join(save_path, 'pred_whole.png'))
    # gt_cls = [cls for cls in gt_cls if cls != 0]
    for cls in gt_cls:
        if cls == 0:
            continue
        class_name = classes[cls]
        # mask_uint8 = 255 * np.uint8(pred_mask == cls)
        # gt_uint8 = 255 * np.uint8(gt_mask == cls)
        # mask_float = np.float32(pred_mask == cls)

        # mmcv.imwrite(gt_uint8, osp.join(save_path, 'gt_' + class_name + '.png'))
        # mmcv.imwrite(mask_uint8, osp.join(save_path, 'pred_' + class_name + '.png'))
        # print(f'pred mask saved to {save_path}')
        # warp the target_category with ClassifierOutputTarget in grad_cam>=1.3.7,
        # to fix the bug in #654.

        # targets = [SemanticSegmentationTarget(cls, mask_float)]
        targets = [SemanticSegmentationTarget(c, np.float32(pred_mask == c)) for c in gt_cls]
        # calculate cam grads and show|save the visualization image

        grayscale_cam = cam(
            data['inputs'],
            targets,
            eigen_smooth=args.eigen_smooth,
            aug_smooth=args.aug_smooth)
        show_cam_grad(
            grayscale_cam,
            src_img,
            title=args.method,
            out_path=osp.join(save_path, 'gradcam_' + 'all' + '.png'),
            return_img=False)

        break


if __name__ == '__main__':
    main()
