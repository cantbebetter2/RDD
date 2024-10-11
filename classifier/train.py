import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import torch.nn.functional as F
import numpy as np
import random
import os
import logging
import sys
from autoaugment import CIFAR10Policy
from cutout import Cutout
import utils as utils
from tensorboardX import SummaryWriter
import torch.distributed as dist
import importlib
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Baseline')
parser.add_argument('--model', default="resnet18", type=str)
parser.add_argument('--dataset', default="cifar10", type=str, help="cifar10")
parser.add_argument('--gpu_id', default='0', type=str, help='id for CUDA_VISIBLE_DEVICES')
parser.add_argument('--epoch', default=300, type=int, help="training epochs")
parser.add_argument('--seed', default=-1, type=int)
parser.add_argument('--result_path', default="./result/", type=str)
parser.add_argument('--batchsize', default=128, type=int)
parser.add_argument('--init_lr', default=0.1, type=float)
parser.add_argument('--wd', default=5e-4, type=float)
parser.add_argument('--normalize05', action='store_true')
args = parser.parse_args()

model_dir = args.result_path+args.dataset+'/'+args.model
if args.normalize05:
    model_dir += '_normalize05'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
utils.set_logger(os.path.join(model_dir, 'train.log'))
logging.info(args)
writer = SummaryWriter(log_dir = model_dir)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True

if args.seed != -1:
    setup_seed(args.seed)

big_size = False
if args.normalize05:
    norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
else:
    norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
if args.dataset == "cifar10":
    transform_train = transforms.Compose(
        [transforms.RandomCrop(32, padding=4, fill=128),
        transforms.RandomHorizontalFlip(), CIFAR10Policy(), 
        transforms.ToTensor(), Cutout(n_holes=1, length=16), 
        norm])
    transform_test = transforms.Compose(
        [transforms.ToTensor(), 
        norm])
    
if args.dataset == "cifar10":
    trainset = torchvision.datasets.CIFAR10(root="../../../data", train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root="../../../data", train=False, download=True, transform=transform_test)
    num_class = 10

trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=args.batchsize,
    shuffle=True,
    num_workers=4
)
testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=args.batchsize,
    shuffle=False,
    num_workers=4
)

if args.model.startswith('res') or args.model.startswith('wide'):
    M = importlib.import_module("resnet")
    net = getattr(M, args.model)(num_classes=num_class, is_big_size=big_size)
if args.model.startswith('densenet'):
    M = importlib.import_module("densenet")
    net = getattr(M, args.model)(num_classes=num_class, is_big_size=big_size)

if __name__ == "__main__":
    net = net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.init_lr, weight_decay=args.wd, momentum=0.9)

    best_final_acc = 0
    for epoch in range(args.epoch):
        correct = 0
        predicted = 0
        if epoch in [args.epoch // 2, args.epoch * 3 // 4]:
            for param_group in optimizer.param_groups:
                param_group['lr'] /= 10

        net.train()
        sum_loss, total = 0.0, 0.0
        for i, data in enumerate(trainloader, 0):
            length = len(trainloader)
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            output, _ = net(inputs)

            loss = torch.FloatTensor([0.]).to(device)

            loss += criterion(output, labels)

            sum_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(output.data, 1)
            correct += float(predicted.eq(labels.data).cpu().sum())
            total += float(labels.size(0))
        writer.add_scalar('Train_Acc', 100 * correct / total, epoch + 1)
        writer.add_scalar('Train_Loss', sum_loss / total, epoch + 1)

        with torch.no_grad():
            correct = 0
            predicted = 0
            net.eval()
            total = 0.0
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                output, _ = net(images)
                _, predicted = torch.max(output.data, 1)
                correct += float(predicted.eq(labels.data).cpu().sum())
                total += float(labels.size(0))

            logging.info('Epoch {}, Acc: {:.2f}%'.format(epoch + 1, 100 * correct / total))
            if 100 * correct / total > best_final_acc:
                best_final_acc = 100 * correct / total
                logging.info("Best Final Accuracy Updated: {:.2f}%".format(best_final_acc))
                ckpt = {'model': net.state_dict()}
                torch.save(ckpt, os.path.join(model_dir, 'ckpt.pt'))
        writer.add_scalar('Final_Acc', 100 * correct / total, epoch + 1)
        writer.add_scalar('Best_Final_Acc', best_final_acc, epoch + 1)

    writer.close()
    logging.info("Training Finished, Total EPOCH={}, Best Final Accuracy={:.2f}%".format(args.epoch, best_final_acc))
