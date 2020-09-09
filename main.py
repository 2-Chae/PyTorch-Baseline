import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

import argparse
import os

from utils import *
from utils.misc import _create_info_folder, progress_bar


parser = argparse.ArgumentParser(description='PyTorch Training Baseline')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--weight-decay', type=float, default=5e-4, help='weight decay')
parser.add_argument('--epochs', type=int, default=164, help='# of epochs to train')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--save-image', action='store_true', default=False, help='save input image')
parser.add_argument('--batch-train', type=int, default=128, help='batch size for train set')
parser.add_argument('--batch-test', type=int, default=128, help='batch size for test set')
args = parser.parse_args()

print(args)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data augmentation
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


# define dataset
trainset = torchvision.datasets.CIFAR10(root='~/Downloads', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_train, shuffle=True, num_workers=4)

testset = torchvision.datasets.CIFAR10(root='~/Downloads', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_test, shuffle=False, num_workers=4)


# Model
net = models.resnet18(pretrained=True)
net = net.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True  


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[82, 123], gamma=0.1)

# log
board_writer = SummaryWriter()
_create_info_folder(board_writer, args, files_to_same=["main.py"])
model_checkpoints_ = os.path.join(board_writer.log_dir, 'checkpoints', 'model.pth')
csv_logger = CSVLogger(fieldnames=['epoch', 'train_acc', 'train_loss', 'test_acc'], filename=os.path.join(board_writer.log_dir, 'log.csv'))


def test(dataloader):
    net.eval()

    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    test_acc = 100.*correct/total
    return test_acc

# Training
n_iter = 0
for epoch in range(args.epochs):
    print('\n[Epoch: %d]' % epoch)
    net.train()

    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        if args.save_image and batch_idx == 0:
            grid = torchvision.utils.make_grid(inputs[:32])
            board_writer.add_image('input_image', grid, global_step=n_iter)

        outputs = net(inputs)
        loss = criterion(outputs, targets)
        board_writer.add_scalar('loss', loss, global_step=n_iter)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

        n_iter += 1

    test_acc = test(testloader)
    print('test_acc: %.3f' % test_acc)
    scheduler.step()

    row = {'epoch': str(epoch), 'train_acc': str(100.*correct/total), 'train_loss': str(train_loss/(batch_idx+1)), 'test_acc': str(test_acc)}
    csv_logger.writerow(row)

# save model checkpoint
torch.save({
    'net': net.state_dict(),
}, model_checkpoints_)