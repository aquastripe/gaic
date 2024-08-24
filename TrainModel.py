import argparse
import math
import os
import random
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.optim as optim
from scipy.stats import spearmanr, pearsonr
from torch.utils.data import DataLoader
from tqdm.rich import tqdm

from croppingDataset import GAICD
from croppingModel import build_crop_model

SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


def parse_args():
    parser = argparse.ArgumentParser(description='Grid anchor based image cropping')
    parser.add_argument('--dataset_root', default='dataset/GAIC/', help='Dataset root directory path')
    parser.add_argument('--base_model', default='mobilenetv2', help='Pretrained base model')
    parser.add_argument('--no_rod', action='store_true', default=False, help='No RoD Align')
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


def collate_gaicd(batch):
    images = []
    bbox = defaultdict(list)
    mos = []
    for item in batch:
        images.append(item['image'])
        bbox['xmin'] += item['bbox']['xmin']
        bbox['xmax'] += item['bbox']['xmax']
        bbox['ymin'] += item['bbox']['ymin']
        bbox['ymax'] += item['bbox']['ymax']
        mos += item['MOS']
    images = torch.stack(images)
    results = {
        'image': images,
        'bbox': bbox,
        'MOS': mos,
    }
    return results


@torch.no_grad()
def test(model, data_loader_test):
    model.eval()
    acc4_5 = []
    acc4_10 = []
    wacc4_5 = []
    wacc4_10 = []
    srcc = []
    pcc = []
    total_loss = 0
    avg_loss = 0
    for n in range(4):
        acc4_5.append(0)
        acc4_10.append(0)
        wacc4_5.append(0)
        wacc4_10.append(0)

    for id, sample in enumerate(tqdm(data_loader_test)):
        image = sample['image']
        bboxs = sample['bbox']
        MOS = sample['MOS']

        roi = []
        for idx in range(0, len(bboxs['xmin'])):
            roi.append((0, bboxs['xmin'][idx], bboxs['ymin'][idx], bboxs['xmax'][idx], bboxs['ymax'][idx]))
        roi = torch.tensor(roi).float()
        MOS = torch.tensor(MOS)

        if torch.cuda.is_available():
            image = image.to('cuda')
            roi = roi.to('cuda')
            MOS = MOS.to('cuda')

        # t0 = time.time()
        out = model(image, roi)
        loss = torch.nn.SmoothL1Loss(reduction='mean')(out.squeeze(), MOS)
        total_loss += loss.item()
        avg_loss = total_loss / (id + 1)

        id_MOS = sorted(range(len(MOS)), key=lambda k: MOS[k], reverse=True)
        id_out = sorted(range(len(out)), key=lambda k: out[k], reverse=True)
        for k in range(4):
            temp_acc_4_5 = 0.0
            temp_acc_4_10 = 0.0
            for j in range(k + 1):
                if MOS[id_out[j]] >= MOS[id_MOS[4]]:
                    temp_acc_4_5 += 1.0
                if MOS[id_out[j]] >= MOS[id_MOS[9]]:
                    temp_acc_4_10 += 1.0
            acc4_5[k] += temp_acc_4_5 / (k + 1.0)
            acc4_10[k] += temp_acc_4_10 / (k + 1.0)

        rank_of_returned_crop = []
        for k in range(4):
            rank_of_returned_crop.append(id_MOS.index(id_out[k]))

        for k in range(4):
            temp_wacc_4_5 = 0.0
            temp_wacc_4_10 = 0.0
            temp_rank_of_returned_crop = rank_of_returned_crop[:(k + 1)]
            temp_rank_of_returned_crop.sort()
            for j in range(k + 1):
                if temp_rank_of_returned_crop[j] <= 4:
                    temp_wacc_4_5 += 1.0 * math.exp(-0.2 * (temp_rank_of_returned_crop[j] - j))
                if temp_rank_of_returned_crop[j] <= 9:
                    temp_wacc_4_10 += 1.0 * math.exp(-0.1 * (temp_rank_of_returned_crop[j] - j))
            wacc4_5[k] += temp_wacc_4_5 / (k + 1.0)
            wacc4_10[k] += temp_wacc_4_10 / (k + 1.0)

        MOS_arr = MOS.cpu().numpy()
        out = torch.squeeze(out).cpu().numpy()
        srcc.append(spearmanr(MOS_arr, out).statistic)
        pcc.append(pearsonr(MOS_arr, out).statistic)

        # t1 = time.time()

        # print('timer: %.4f sec.' % (t1 - t0))
    for k in range(4):
        acc4_5[k] = acc4_5[k] / 200.0
        acc4_10[k] = acc4_10[k] / 200.0
        wacc4_5[k] = wacc4_5[k] / 200.0
        wacc4_10[k] = wacc4_10[k] / 200.0

    avg_srcc = sum(srcc) / 200.0
    avg_pcc = sum(pcc) / 200.0

    return acc4_5, acc4_10, avg_srcc, avg_pcc, avg_loss, wacc4_5, wacc4_10


def train(model, data_loader_train, data_loader_test, optimizer, args):
    model.train()
    for epoch in range(0, 80):
        total_loss = 0
        for id, sample in enumerate(data_loader_train):
            image = sample['image']
            bboxs = sample['bbox']

            roi = []
            MOS = []

            random_ID = list(range(0, len(bboxs['xmin'])))
            random.shuffle(random_ID)

            for idx in random_ID[:64]:
                roi.append((0, bboxs['xmin'][idx], bboxs['ymin'][idx], bboxs['xmax'][idx], bboxs['ymax'][idx]))
                MOS.append(sample['MOS'][idx])
            roi = torch.tensor(roi).float()
            MOS = torch.tensor(MOS)

            if torch.cuda.is_available():
                image = image.to('cuda')
                roi = roi.to('cuda')
                MOS = MOS.to('cuda')

            # forward
            out = model(image, roi)
            loss = torch.nn.SmoothL1Loss(reduction='mean')(out.squeeze(), MOS)
            total_loss += loss.item()
            avg_loss = total_loss / (id + 1)

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sys.stdout.write(
                '\r[Epoch %d/%d] [Batch %d/%d] [Train Loss: %.4f]' % (epoch, 79, id, len(data_loader_train), avg_loss))

        acc4_5, acc4_10, avg_srcc, avg_pcc, test_avg_loss, wacc4_5, wacc4_10 = test(model, data_loader_test)
        sys.stdout.write(
            '[Test Loss: %.4f] [%.3f, %.3f, %.3f, %.3f] [%.3f, %.3f, %.3f, %.3f] [SRCC: %.3f] [PCC: %.3f]\n' % (
                test_avg_loss, acc4_5[0], acc4_5[1], acc4_5[2], acc4_5[3], acc4_10[0], acc4_10[1], acc4_10[2],
                acc4_10[3],
                avg_srcc, avg_pcc))
        sys.stdout.write('[%.3f, %.3f, %.3f, %.3f] [%.3f, %.3f, %.3f, %.3f]\n' % (
            wacc4_5[0], wacc4_5[1], wacc4_5[2], wacc4_5[3], wacc4_10[0], wacc4_10[1], wacc4_10[2], wacc4_10[3]))
        file_path = f'{args.save_folder}/{epoch:02d}_{avg_srcc:.3f}.pth'
        torch.save(model.state_dict(), file_path)


def main():
    args = parse_args()
    folder_name = (f'downsample_{args.downsample}-{args.scale}-Aug_{args.augmentation}-Align_{args.align_size}-'
                f'Cdim_{args.reduced_dim}')
    if args.no_rod:
        folder_name += '-no_rod'

    args.save_folder = args.save_folder + args.base_model + '/' + folder_name

    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    train_set = GAICD(image_size=args.image_size, dataset_dir=args.dataset_root, set='train',
                      augmentation=args.augmentation)
    data_loader_train = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True,
                                   worker_init_fn=random.seed(SEED), persistent_workers=True, collate_fn=collate_gaicd)
    test_set = GAICD(image_size=args.image_size, dataset_dir=args.dataset_root, set='test')
    data_loader_test = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False,
                                  persistent_workers=True, collate_fn=collate_gaicd)

    net = build_crop_model(scale=args.scale, alignsize=args.align_size, reddim=args.reduced_dim, loadweight=True,
                           model=args.base_model, downsample=args.downsample, no_rod=args.no_rod)

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # cudnn.benchmark = True
        net = net.cuda()

    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    train(net, data_loader_train, data_loader_test, optimizer, args)


if __name__ == '__main__':
    main()
