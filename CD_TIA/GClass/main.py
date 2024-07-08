import os
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from tqdm import tqdm
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from net import ResNet


def train_and_valid(model, loss_function, optimizer, epochs=25):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    history = []
    best_acc = 0.0
    best_epoch = 0

    for epoch in range(epochs):
        epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch + 1, epochs))

        # model.train()

        train_loss = 0.0
        train_acc = 0.0
        valid_loss = 0.0
        valid_acc = 0.0
        train_FN = 0
        train_FP = 0
        train_TP = 0
        train_TN = 0
        valid_FN = 0
        valid_FP = 0
        valid_TP = 0
        valid_TN = 0
        for k, res_list in enumerate(tqdm(train_data)):
            inputs = res_list[0].to(device)
            labels = res_list[1].to(device).long()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            train_acc += acc.item() * inputs.size(0)

            for i in range(len(predictions)):
                if predictions[i] == labels[i] and labels[i] == 0:
                    train_TP += 1
                elif predictions[i] == labels[i] and labels[i] == 1:
                    train_TN += 1
                elif predictions[i] == 1 and labels[i] == 0:
                    train_FN += 1
                elif predictions[i] == 0 and labels[i] == 1:
                    train_FP += 1

        with torch.no_grad():
            model.eval()

            for j, rs_list in enumerate(tqdm(valid_data)):
                inputs = rs_list[0].to(device)
                labels = rs_list[1].to(device)
                outputs = model(inputs)
                #print(labels.shape)
                loss = loss_function(outputs, labels)
                valid_loss += loss.item() * inputs.size(0)
                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))
                acc = torch.mean(correct_counts.type(torch.FloatTensor))
                valid_acc += acc.item() * inputs.size(0)

                for i in range(len(predictions)):
                    if predictions[i] == labels[i] and labels[i] == 0:
                        valid_TP += 1
                    elif predictions[i] == labels[i] and labels[i] == 1:
                        valid_TN += 1
                    elif predictions[i] == 1 and labels[i] == 0:
                        valid_FN += 1
                    elif predictions[i] == 0 and labels[i] == 1:
                        valid_FP += 1

        avg_train_loss = train_loss / train_data_size
        avg_train_acc = train_acc / train_data_size

        avg_valid_loss = valid_loss / valid_data_size
        avg_valid_acc = valid_acc / valid_data_size

        history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])

        if best_acc < avg_valid_acc:
            best_acc = avg_valid_acc
            best_epoch = epoch + 1

        epoch_end = time.time()
        print(
            "Epoch: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n\t\tValidation: Loss: {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(
                epoch + 1, avg_valid_loss, avg_train_acc * 100, avg_valid_loss, avg_valid_acc * 100,
                epoch_end - epoch_start
            ))

        print("Epoch:{:03d}, train_data_size: {:03d}, valid_data_size:{:03d}, \n\t\t "
              "train_TP: {:03d}, train_TN: {:03d}, train_FP: {:03d}, train_FN: {:03d},\n\t\t"
              "valid_TP: {:03d}, valid_TN: {:03d}, valid_FP: {:03d}, valid_FN: {:03d},".format(epoch + 1,
                                                                                               train_data_size,
                                                                                               valid_data_size,
                                                                                               train_TP, train_TN,
                                                                                               train_FP, train_FN,
                                                                                               valid_TP, valid_TN,
                                                                                               valid_FP, valid_FN))

        print("Best Accuracy for validation : {:.4f} at epoch {:03d}".format(best_acc, best_epoch))

        torch.save(model.state_dict(), 'OnlyChangModel/' + dataset + '_model_' + str(epoch + 1) + '.pth')
    return model, history


if __name__ == '__main__':

    image_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    }


    dataset = 'data'
    # pretrained_file = "model/resnet50-5c106cde.pth"

    '''isLocal = False
    if isLocal:
        datasets = "../" + str(datasets)
        pretrained_file = "../" + str(pretrained_file)'''



    train_directory = os.path.join(dataset, 'train')
    valid_directory = os.path.join(dataset, 'valid')

    batch_size = 8
    data = {
        'train': datasets.ImageFolder(root=train_directory, transform=image_transforms['train']),
        'valid': datasets.ImageFolder(root=valid_directory, transform=image_transforms['valid'])
    }

    train_data_size = len(data['train'])
    valid_data_size = len(data['valid'])
    #print(data['train'])
    train_data = DataLoader(data['train'], batch_size=batch_size, shuffle=True, num_workers=4)
    valid_data = DataLoader(data['valid'], batch_size=batch_size, shuffle=True, num_workers=4)

    resnet50 =ResNet()
    #resnet50=DeepLabV3()


    #resnet50 =DeepLab(2, backbone="mobilenet", pretrained=False, downsample_factor=16)

    # resnet50.load_state_dict(torch.load(pretrained_file))

    # resnet50 = models.resnet50(pretrained=True)



    # for param in resnet50.parameters():
    #     param.requires_grad = False

    # fc_inputs = resnet50.fc.in_features
    # resnet50.fc = nn.Sequential(
    #     nn.Linear(fc_inputs, 256),
    #     nn.ReLU(),
    #     nn.Dropout(0.4),
    #     nn.Linear(256, 2),
    #     nn.LogSoftmax(dim=1)
    # )

    if torch.cuda.is_available():
        resnet50 = resnet50.to('cuda:0')
        print("*" * 20, "Use GPU！")
    else:
        print("*" * 20, "Use CPU！")

   # loss_func = nn.NLLLoss()
    loss_func=nn.CrossEntropyLoss()
    optimizer = optim.Adam(resnet50.parameters())

   # optimizer = optim.SGD(resnet50.parameters())
    num_epochs =200

    trained_model, history = train_and_valid(resnet50, loss_func, optimizer, num_epochs)
    print(trained_model)
    torch.save(history, 'models/' + dataset + '_history.pt')

    history = np.array(history)
    plt.plot(history[:, 0:2])
    plt.legend(['Tr Loss', 'Val Loss'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    plt.ylim(0, 1)
    plt.savefig(dataset + '_loss_curve.png')

    plt.plot(history[:, 2:4])
    plt.legend(['Tr Accuracy', 'Val Accuracy'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.savefig(dataset + '_accuracy_curve.png')
