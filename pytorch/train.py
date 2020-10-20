# 2020.06.09-Changed for main script for testing GhostNet on ImageNet
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
"""PyTorch Inference Script

An example inference script that outputs top-k class ids for images in a folder into a csv.

Hacked together by Ross Wightman (https://github.com/rwightman)
"""
import os
import time
import argparse
import logging
# import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tqdm import tqdm
from ghostnet import ghostnet
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='PyTorch ImageNet Inference')
parser.add_argument('--data', metavar='DIR', default='../data/imagenet/',
                    help='path to dataset')
parser.add_argument('--output_dir', metavar='DIR', default='../models/',
                    help='path to output files')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--num-classes', type=int, default=1000,
                    help='Number classes in dataset')
parser.add_argument('--width', type=float, default=1.0, 
                    help='Width ratio (default: 1.0)')
parser.add_argument('--dropout', type=float, default=0.2, metavar='PCT',
                    help='Dropout rate (default: 0.2)')
parser.add_argument('--num-gpu', type=int, default=0,
                    help='Number of GPUS to use')
parser.add_argument('--epochs', type=int, default=10,
                    help='Number of epoch to train')
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight-decay', type=float, default=1e-4,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')


def main():
    args = parser.parse_args()

    model = ghostnet(num_classes=args.num_classes, width=args.width, dropout=args.dropout)
    model.load_state_dict(torch.load('./models/state_dict_93.98.pth'))

    if args.num_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpu))).cuda()
    elif args.num_gpu < 1:
        model = model
    else:
        model = model.cuda()
    print('GhostNet created.')
    
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
    loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    model.eval()
    
    validate_loss_fn = nn.CrossEntropyLoss()
    if args.num_gpu >=1:
        validate_loss_fn = nn.CrossEntropyLoss().cuda()
    eval_metrics = validate(model, loader, validate_loss_fn, args)
    print(eval_metrics)


def validate(model, loader, loss_fn, args, log_suffix=''):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            # input = input.cuda()
            # target = target.cuda()

            output = model(input)
            if isinstance(output, (tuple, list)):
                output = output[0]

            loss = loss_fn(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            reduced_loss = loss.data

            # torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if (last_batch or batch_idx % 10 == 0):
                log_name = 'Test' + log_suffix
                logging.info(
                    '{0}: [{1:>4d}/{2}]  '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                    'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})'.format(
                        log_name, batch_idx, last_idx, batch_time=batch_time_m,
                        loss=losses_m, top1=top1_m, top5=top5_m))

    metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])

    return metrics


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [correct[:k].view(-1).float().sum(0) * 100. / batch_size for k in topk]


def train_ghostNet():
    "train ghostNet on cifar-10"
    args = parser.parse_args()
    model = ghostnet(
        num_classes=10,
        width=args.width)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
    train_dataset = datasets.CIFAR10(
        root='../data', train=True, download=True,
        transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ]))
    model.train()
    # optimizer = torch.optim.Adam(model.parameters(), 0.001)
    optimizer = torch.optim.SGD(
        model.parameters(), args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay)
    epochs = args.epochs
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join("./runs", current_time)
    writer = SummaryWriter(log_dir)
    for epoch in range(epochs):
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True
            )
        for step, batch in enumerate(tqdm(train_loader)):
            inputs = batch[0]
            target = batch[1]
            output = model(inputs)
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("epoch: {}| loss：{}".format(epoch, loss.data))
            writer.add_scalar('loss', loss, step)


def train_resnet():
    "train resnet on cifar-10"
    args = parser.parse_args()
    from resnet import resnet56
    model = resnet56()
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
    train_dataset = datasets.CIFAR10(
        root='../data', train=True, download=True,
        transform=transforms.Compose([
            # transforms.Resize(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32),
            transforms.ToTensor(),
            normalize
        ]))
    val_dataset = datasets.CIFAR10(
        root='../data', train=False, download=True,
        transform=transforms.Compose([
            # transforms.Resize(256),
            # transforms.CenterCrop(32),
            transforms.ToTensor(),
            normalize
        ]))
    # optimizer = torch.optim.Adam(model.parameters(), 0.001)
    optimizer = torch.optim.SGD(
        model.parameters(), args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[100, 150])
    epochs = args.epochs
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join("./runs", current_time)
    writer = SummaryWriter(log_dir)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        print("epoch: ", epoch)
        model.train()
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True
            )
        for step, batch in enumerate(tqdm(train_loader)):
            inputs = batch[0]
            target = batch[1]
            output = model(inputs)
            loss = loss_fn(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print("epoch: {}| loss：{}".format(epoch, loss.data))
            writer.add_scalar('loss', loss, step)
        lr_scheduler.step()
        # evaluation
        model.eval()
        with torch.no_grad():
            val_loader = torch.utils.data.DataLoader(
                dataset=val_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.workers,
                pin_memory=True
                )
            prec1 = 0
            num = len(val_loader)
            for step, batch in enumerate(tqdm(val_loader)):
                inputs = batch[0]
                target = batch[1]
                output = model(inputs)
                prec1 += accuracy(output.data, target)[0]
            print("epoch: {}| loss：{}".format(epoch, prec1/num))
            writer.add_scalar('avg accuracy', prec1/num, epoch)


if __name__ == '__main__':
    # main()
    train_resnet()