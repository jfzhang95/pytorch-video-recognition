import timeit
from datetime import datetime
import socket
import os
import glob

import torch
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloaders.dataset import VideoDataset
from network import C3D_model, R2Plus1D_model

# Use GPU if available else revert to CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device being used:", device)

nEpochs = 200  # Number of epochs for training
resume_epoch = 0  # Default is 0, change if want to resume
useTest = True # See evolution of the test set when training
nTestInterval = 40 # Run on test set every nTestInterval epochs
snapshot = 40 # Store a model every snapshot epochs
lr = 1e-2 # Learning rate

dataset = 'ucf101'

if dataset == 'hmdb51':
    num_classes=51
elif dataset == 'ucf101':
    num_classes = 101
else:
    raise NotImplementedError

save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
exp_name = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]

if resume_epoch != 0:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) if runs else 0
else:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0

save_dir = os.path.join(save_dir_root, 'run', 'run_' + str(run_id))
modelName = 'C3D' # options: C3D or R2Plus1D


def train_model(dataset=dataset, save_dir=save_dir, num_classes=num_classes, lr=lr,
                num_epochs=nEpochs, save_epoch=snapshot, useTest=useTest, test_interval=nTestInterval):
    """
        Args:
            num_classes (int): Number of classes in the data
            num_epochs (int, optional): Number of epochs to train for.
    """

    if modelName == 'C3D':
        model = C3D_model.C3D(num_classes=num_classes, pretrained=True)
        train_params = [{'params': C3D_model.get_1x_lr_params(model), 'lr': lr},
                        {'params': C3D_model.get_10x_lr_params(model), 'lr': lr * 10}]
    elif modelName == 'R2Plus1D':
        model = R2Plus1D_model.R2Plus1DClassifier(num_classes=num_classes, layer_sizes=(2, 2, 2, 2))
        train_params = [{'params': R2Plus1D_model.get_1x_lr_params(model), 'lr': lr},
                        {'params': R2Plus1D_model.get_10x_lr_params(model), 'lr': lr * 10}]
    else:
        raise NotImplementedError
    criterion = nn.CrossEntropyLoss()  # standard crossentropy loss for classification
    optimizer = optim.SGD(train_params, lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10,
                                          gamma=0.1)  # the scheduler divides the lr by 10 every 10 epochs

    if resume_epoch == 0:
        print("Training {} from scratch...".format(modelName))
    else:
        checkpoint = torch.load(os.path.join(save_dir, 'models', modelName + '_epoch-' + str(resume_epoch - 1) + '.pth'),
                       map_location=lambda storage, loc: storage)   # Load all tensors onto the CPU
        print("Initializing weights from: {}...".format(
            os.path.join(save_dir, 'models', modelName + '_epoch-' + str(resume_epoch - 1) + '.pth')))
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['opt_dict'])
    model.to(device)

    log_dir = os.path.join(save_dir, 'models', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    writer = SummaryWriter(log_dir=log_dir)

    print('Training model on {} dataset...'.format(dataset))
    train_dataloader = DataLoader(VideoDataset(dataset=dataset, split='train', clip_len=16), batch_size=16, shuffle=True, num_workers=4)
    val_dataloader   = DataLoader(VideoDataset(dataset=dataset, split='val', clip_len=16), batch_size=16, shuffle=True, num_workers=4)
    test_dataloader  = DataLoader(VideoDataset(dataset=dataset, split='test', clip_len=16), batch_size=16, num_workers=4)

    trainval_loaders = {'train': train_dataloader, 'val': val_dataloader}
    trainval_sizes = {x: len(trainval_loaders[x].dataset) for x in ['train', 'val']}
    test_size = len(test_dataloader.dataset)

    for epoch in range(resume_epoch, num_epochs):
        # each epoch has a training and validation step
        for phase in ['train', 'val']:
            start_time = timeit.default_timer()

            # reset the running loss and corrects
            running_loss = 0.0
            running_corrects = 0

            # set model to train() or eval() mode depending on whether it is trained
            # or being validated. Primarily affects layers such as BatchNorm or Dropout.
            if phase == 'train':
                # scheduler.step() is to be called once every epoch during training
                scheduler.step()
                model.train()
            else:
                model.eval()

            for inputs, labels in tqdm(trainval_loaders[phase]):
                # move inputs and labels to the device the training is taking place on
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                # keep intermediate states iff backpropagation will be performed. If false,
                # then all intermediate states will be thrown away during evaluation, to use
                # the least amount of memory possible.
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    preds = torch.max(outputs, 1)[1]
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / trainval_sizes[phase]
            epoch_acc = running_corrects.double() / trainval_sizes[phase]

            if phase == 'train':
                writer.add_scalar('data/train_loss_epoch', epoch_loss, epoch)
                writer.add_scalar('data/train_acc_epoch', epoch_acc, epoch)
            else:
                writer.add_scalar('data/val_loss_epoch', epoch_loss, epoch)
                writer.add_scalar('data/val_acc_epoch', epoch_acc, epoch)

            print("[{}] Epoch: {}/{} Loss: {} Acc: {}".format(phase, epoch+1, nEpochs, epoch_loss, epoch_acc))
            stop_time = timeit.default_timer()
            print("Execution time: " + str(stop_time - start_time) + "\n")

        if epoch % save_epoch == (save_epoch - 1):
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'opt_dict': optimizer.state_dict(),
            }, os.path.join(save_dir, 'models', modelName + '_epoch-' + str(epoch) + '.pth.tar'))
            print("Save model at {}\n".format(os.path.join(save_dir, 'models', modelName + '_epoch-' + str(epoch) + '.pth.tar')))

        if useTest and epoch % test_interval == (test_interval - 1):
            start_time = timeit.default_timer()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(test_dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.no_grad():
                    outputs = model(inputs)
                preds = torch.max(outputs, 1)[1]
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / test_size
            epoch_acc = running_corrects.double() / test_size

            writer.add_scalar('data/test_loss_epoch', epoch_loss, epoch)
            writer.add_scalar('data/test_acc_epoch', epoch_acc, epoch)

            print("[test] Epoch: {}/{} Loss: {} Acc: {}".format(epoch+1, nEpochs, epoch_loss, epoch_acc))
            stop_time = timeit.default_timer()
            print("Execution time: " + str(stop_time - start_time) + "\n")

    writer.close()


if __name__ == "__main__":
    train_model()