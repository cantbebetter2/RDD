import torch
import torchvision
import torchvision.transforms as transforms
import argparse
import numpy as np
import os
import warnings
from inception import InceptionV3
from torchvision.datasets import ImageFolder, STL10
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Baseline')
parser.add_argument('--dataset', default="imagenet64", type=str, help="cifar10")
parser.add_argument('--gpu_id', default='4', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--batchsize', default=128, type=int)
parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_images = 0
if args.dataset == "cifar10":
    transform_train = transforms.Compose([transforms.ToTensor()])
    num_images = 50000
    dataset = torchvision.datasets.CIFAR10(root="/data/dataset", train=True, download=False, transform=transform_train)

elif args.dataset == "imagenet64":
    dataset = ImageFolder('/data/fwl/dataset/imagenet64/train', transforms.Compose([transforms.ToTensor(),
                                                                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    num_images = 1281167


loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=args.batchsize,
    shuffle=False,
    num_workers=4
)

if __name__ == "__main__":
    block_idx1 = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = InceptionV3([block_idx1]).to(device)
    model.eval()
    
    fid_acts = np.empty((num_images, 2048))
    start = 0
    with torch.no_grad():
        for data in loader:
            images, _ = data
            images = images.to(device)
            end = start + len(images)
            pred = model(images)[0]
            fid_acts[start: end] = pred.view(-1, 2048).cpu().numpy()
            start = end
            print(end)
    m1 = np.mean(fid_acts, axis=0)
    s1 = np.cov(fid_acts, rowvar=False)
    np.savez('./stats/'+args.dataset+'.train.npz', mu=m1, sigma=s1)