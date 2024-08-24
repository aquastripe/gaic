import argparse
import os

import torch
from torch.utils.data import DataLoader

from TrainModel import collate_gaicd, test
from croppingDataset import GAICD
from croppingModel import build_crop_model

SEED = 0


def parse_args():
    parser = argparse.ArgumentParser(description='Grid anchor based image cropping')
    parser.add_argument('--dataset_root', default='dataset/GAIC/', help='Dataset root directory path')
    parser.add_argument('--base_model', default='mobilenetv2', help='Pretrained base model')
    parser.add_argument('--params', default=None, help='Model parameters for testing')
    parser.add_argument('--scale', default='multi', type=str, help='choose single or multi scale')
    parser.add_argument('--downsample', default=4, type=int, help='downsample time')
    parser.add_argument('--augmentation', default=1, type=int, help='choose single or multi scale')
    parser.add_argument('--image_size', default=256, type=int, help='Batch size for training')
    parser.add_argument('--align_size', default=9, type=int, help='Spatial size of RoIAlign and RoDAlign')
    parser.add_argument('--reduced_dim', default=8, type=int, help='Spatial size of RoIAlign and RoDAlign')
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size for training')
    parser.add_argument('--resume', default=None, type=str, help='Checkpoint state_dict file to resume training from')
    parser.add_argument('--start_iter', default=0, type=int, help='Resume training at this iter')
    parser.add_argument('--num_workers', default=0, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='initial learning rate')
    parser.add_argument('--save_folder', default='weights/ablation/cropping/',
                        help='Directory for saving checkpoint models')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    args.save_folder = args.save_folder + args.base_model + '/' + 'downsample' + str(
        args.downsample) + '_' + args.scale + '_Aug' + str(args.augmentation) + '_Align' + str(
        args.align_size) + '_Cdim' + str(args.reduced_dim)

    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    test_set = GAICD(image_size=args.image_size, dataset_dir=args.dataset_root, set='test')
    data_loader_test = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False,
                                  persistent_workers=True, collate_fn=collate_gaicd)
    net = build_crop_model(scale=args.scale, alignsize=args.align_size, reddim=args.reduced_dim, loadweight=True,
                           model=args.base_model, downsample=args.downsample)
    params = torch.load(args.params, map_location='cpu')
    net.load_state_dict(params)

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # cudnn.benchmark = True
        net = net.cuda()

    acc4_5, acc4_10, avg_srcc, avg_pcc, test_avg_loss, wacc4_5, wacc4_10 = test(net, data_loader_test)
    print('[Test Loss: {:.4f}] '
          '[{:.3f}, {:.3f}, {:.3f}, {:.3f}] '
          '[{:.3f}, {:.3f}, {:.3f}, {:.3f}] '
          '[SRCC: {:.3f}] [PCC: {:.3f}]'.format(test_avg_loss, *acc4_5, *acc4_10, avg_srcc, avg_pcc))
    print('[{:.3f}, {:.3f}, {:.3f}, {:.3f}] [{:.3f}, {:.3f}, {:.3f}, {:.3f}]'.format(*wacc4_5, *wacc4_10))


if __name__ == '__main__':
    main()
