import argparse
from time import time

import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model import CNN, BestSmallModel, BestModel, CNN2layer, CNN1layer, CNN4layer

from scipy.signal import savgol_filter
from torchsummary import summary


def imshow(img, class_name):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title("Letter: {}".format(class_name))
    plt.show()

    return


def load_data(data_dir, val_data_dir, batch_size):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0, 0, 0), (1, 1, 1))])
    dataset = torchvision.datasets.ImageFolder(data_dir, transform=transform)
    dataset_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    val_dataset = torchvision.datasets.ImageFolder(val_data_dir, transform=transform)
    val_dataset_loader = DataLoader(dataset=val_dataset, batch_size=val_dataset.__len__(), shuffle=True)

    return dataset_loader, val_dataset_loader


def load_model(lr, use_bn, num_fc1_neurons, num_conv_kernels, num_conv=None, run_best=None):

    if not run_best:
        if not num_conv:
            model = CNN(use_bn=use_bn, num_fc1_neurons=num_fc1_neurons, num_conv_kernels=num_conv_kernels)
        elif num_conv == 2:
            model = CNN2layer(use_bn=use_bn, num_fc1_neurons=num_fc1_neurons, num_conv_kernels=num_conv_kernels)
        elif num_conv == 1:
            model = CNN1layer(use_bn=use_bn, num_fc1_neurons=num_fc1_neurons, num_conv_kernels=num_conv_kernels)
        elif num_conv == 4:
            model = CNN4layer(use_bn=use_bn, num_fc1_neurons=num_fc1_neurons, num_conv_kernels=num_conv_kernels)

    elif run_best == 'best':
        model = BestModel(use_bn=use_bn)
    elif run_best == 'small':
        model = BestSmallModel(use_bn=use_bn)

    if args.use_ce:
        loss_fnc = torch.nn.CrossEntropyLoss()
    else:
        loss_fnc = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    return model, loss_fnc, optimizer


def main(args):
    torch.manual_seed(1000)

    # loading data
    dataset, val_dataset = load_data(args.train_datadir, args.val_datadir, args.batch_size)

    # printing 4 images to test if the dataloader works correctly
    '''
    dataiter = iter(dataset)
    images, labels = dataiter.next()

    classes = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K')
    for i in range(4):
        class_name = classes[labels[i].item()]
        imshow(torchvision.utils.make_grid(images[i]), class_name)
    '''
    trainLoss = []
    trainAcc = []
    trainTime = []
    valAcc = []
    valLoss = []

    model, loss_func, optimizer = load_model(args.lr, args.use_bn, args.num_fc1_neurons, args.num_conv_kernels, args.num_conv, args.run_best)

    a = time()
    for i in range(args.epochs):
        running_loss = 0.0
        total_corr = 0.0
        total_data = 0.0
        total_steps = 0.0
        for j, data in enumerate(dataset, 0):
            inputs, labels = data

            if not args.use_ce:
                labels = torch.nn.functional.one_hot(labels, 10).float()
            else:
                labels = labels.type(torch.LongTensor)

            optimizer.zero_grad()

            predicts = model(inputs.float())
            train_loss = loss_func(input=predicts.squeeze(), target=labels)
            train_loss.backward()
            optimizer.step()
            running_loss += train_loss.item()
            _, predicted = torch.max(predicts.data, 1)
            if not args.use_ce:
                _, labels = torch.max(labels.data, 1)
            total_corr += (predicted == labels).sum().item()

            total_data += len(labels)
            total_steps += 1

        val_inputs, val_labels = iter(val_dataset).next()
        if not args.use_ce:
            val_labels = torch.nn.functional.one_hot(val_labels, 10).float()
        else:
            val_labels = val_labels.type(torch.LongTensor)
        val_predicts = model(val_inputs.float())
        val_loss = loss_func(input=val_predicts.squeeze(), target=val_labels)
        _, val_predicted = torch.max(val_predicts.data, 1)
        if not args.use_ce:
            _, val_labels = torch.max(val_labels.data, 1)
        val_acc = (val_predicted == val_labels).sum().item() / len(val_labels)

        running_loss /= total_steps
        total_corr /= total_data
        train_time = time() - a

        print("iter: " + str(i) + " cost: " + str(running_loss) + " training accuracy: " + str(total_corr) + " val accuracy: " + str(val_acc) + " training time: " + str(train_time))

        trainLoss.append(running_loss)
        trainAcc.append(total_corr)
        trainTime.append(train_time)
        valLoss.append(val_loss.item())
        valAcc.append(val_acc)
        running_loss = 0.0
        total_corr = 0.0
        total_data = 0.0
        total_steps = 0.0

    b = time()

    torch.save(model.state_dict(), 'MyBestSmall.pt')

    from sklearn.metrics import confusion_matrix

    cf_matrix = confusion_matrix(val_labels.detach().numpy(), val_predicted.detach().numpy())

    print(cf_matrix)

    summary(model, input_size=(3, 56, 56))

    diff = b - a
    print("time training loop take is : %s" % diff)

    plt.figure()
    plt.title("Training Accuracies and Training Loss vs. Number of Epochs")
    plt.plot(np.array(np.arange(len(trainAcc))), trainAcc, color='orange', label='training accuracy')
    plt.plot(np.array(np.arange(len(trainLoss))), trainLoss, color='blue', label='training loss')
    plt.xlabel("Number of Epochs")
    plt.ylabel("Accuracies/Loss")
    plt.legend()
    plt.show()

    plt.figure()
    plt.title("Validation Accuracies and Validation Loss vs. Number of Epochs")
    plt.plot(np.array(np.arange(len(valAcc))), valAcc, color='orange', label='validation accuracy')
    plt.plot(np.array(np.arange(len(valLoss))), valLoss, color='blue', label='validation loss')
    plt.xlabel("Number of Epochs")
    plt.ylabel("Accuracies/Loss")
    plt.legend()
    plt.show()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.1)

    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--train_datadir', type=str,
                        help='Path to the folder of training data that contains 10 other type of folders of data',
                        default="asl_images/train")
    parser.add_argument('--val_datadir', type=str,
                        help='Path to the folder of validation data that contains 10 other type of folders of data',
                        default="asl_images/validation")
    parser.add_argument('--use_ce', type=bool,
                        help='Whether use Cross Entropy as loss function',
                        default=True)
    parser.add_argument('--use_bn', type=bool,
                        help='Whether use batch normalization',
                        default=True)
    parser.add_argument('--num_fc1_neurons', type=int,
                        help='Number of neurons on the first fully connected layers',
                        default=32)
    parser.add_argument('--num_conv_kernels', type=int,
                        help='Number of kernels on the convolutional layers',
                        default=30)
    parser.add_argument('--num_conv', choices=[1, 2, 4],
                        help='Number of convolutional layers',
                        default=None)
    parser.add_argument('--run_best', choices=['best', 'small'],
                        help='Running the best model or the best small model',
                        default=None)

    args = parser.parse_args()

    print("learning rate = ", args.lr)
    print("number of epoch = ", args.epochs)

    main(args)