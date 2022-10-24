from functools import reduce
from operator import __or__
from pickle import FALSE

import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
from torch.nn import functional as F
from utils import *


class AdversarialDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.targets = labels

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index) -> T_co:
        return self.inputs[index], self.targets[index]


class Pseudo_Dataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.targets = labels

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index) -> T_co:
        return self.inputs[index], self.targets[index]


def per_image_standardization(x):
    y = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
    mean = y.mean(dim=1, keepdim=True).expand_as(y)
    std = y.std(dim=1, keepdim=True).expand_as(y)
    adjusted_std = torch.max(std, 1.0 / torch.sqrt(torch.cuda.FloatTensor([x.shape[1] * x.shape[2] * x.shape[3]])))
    y = (y - mean) / adjusted_std
    standarized_input = y.view(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
    return standarized_input


def load_raw_dataset(data_aug, dataset, data_target_dir):
    if dataset == 'cifar10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif dataset == 'cifar100':
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
    elif dataset == 'svhn':
        mean = [x / 255 for x in [127.5, 127.5, 127.5]]
        std = [x / 255 for x in [127.5, 127.5, 127.5]]
    elif dataset == 'tiny-imagenet-200':
        mean = [x / 255 for x in [127.5, 127.5, 127.5]]
        std = [x / 255 for x in [127.5, 127.5, 127.5]]
    elif dataset == 'mnist':
        pass
    else:
        assert False, "Unknown dataset : {}".format(dataset)

    if data_aug == 1:
        print('data aug')
        if dataset == 'svhn':
            train_transform = transforms.Compose(
                [transforms.RandomCrop(32, padding=2), transforms.ToTensor(),
                 transforms.Normalize(mean, std)])
            test_transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean, std)])
        elif dataset == 'mnist':
            hw_size = 32
            train_transform = transforms.Compose([
                transforms.Resize((hw_size, hw_size)),
                transforms.ToTensor()
            ])
            test_transform = transforms.Compose([
                transforms.Resize((hw_size, hw_size)),
                transforms.ToTensor()
            ])
        elif dataset == 'tiny-imagenet-200':
            train_transform = transforms.Compose(
                [transforms.RandomHorizontalFlip(),
                 transforms.RandomCrop(64, padding=4),
                 transforms.ToTensor(),
                 transforms.Normalize(mean, std)])
            test_transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean, std)])
        else:
            train_transform = transforms.Compose(
                [transforms.RandomHorizontalFlip(),
                 transforms.RandomCrop(32, padding=2),
                 transforms.ToTensor(),
                 transforms.Normalize(mean, std)])
            test_transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean, std)])
    else:
        print('no data aug')
        if dataset == 'mnist':
            hw_size = 32
            train_transform = transforms.Compose([
                transforms.Resize((hw_size, hw_size)),
                transforms.ToTensor()
            ])
            test_transform = transforms.Compose([
                transforms.Resize((hw_size, hw_size)),
                transforms.ToTensor()
            ])
        else:
            train_transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize(mean, std)])
            test_transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean, std)])
    if dataset == 'cifar10':
        train_data = datasets.CIFAR10(data_target_dir, train=True, transform=train_transform, download=True)
        test_data = datasets.CIFAR10(data_target_dir, train=False, transform=test_transform, download=True)
        num_classes = 10
    elif dataset == 'cifar100':
        train_data = datasets.CIFAR100(data_target_dir, train=True, transform=train_transform, download= False)
        test_data = datasets.CIFAR100(data_target_dir, train=False, transform=test_transform, download=False)
        num_classes = 100
    elif dataset == 'svhn':
        train_data = datasets.SVHN(data_target_dir, split='train', transform=train_transform, download=True)
        test_data = datasets.SVHN(data_target_dir, split='test', transform=test_transform, download=True)
        num_classes = 10
    elif dataset == 'mnist':
        train_data = datasets.MNIST(data_target_dir, train=True, transform=train_transform, download=True)
        test_data = datasets.MNIST(data_target_dir, train=False, transform=test_transform, download=True)
        num_classes = 10
    elif dataset == 'tiny-imagenet-200':
        train_root = os.path.join(data_target_dir, 'train')  # this is path to training images folder
        validation_root = os.path.join(data_target_dir, 'val/images')  # this is path to validation images folder
        train_data = datasets.ImageFolder(train_root, transform=train_transform)
        test_data = datasets.ImageFolder(validation_root, transform=test_transform)
        num_classes = 200
    else:
        assert False, "Dataset {} is unsupported.".format(dataset)

    return train_data, test_data, num_classes


def attack_single_batch_input(net, images, labels, num_iter=7, eps=8 / 255, alpha=2/255,  random_start=True):

    images = images.cuda()
    labels = torch.tensor(labels).cuda()
    loss_function = nn.CrossEntropyLoss()

    ori_images = images.data

    if random_start:
        ori_images = ori_images + torch.Tensor(np.random.uniform(-eps, eps, ori_images.shape)).cuda()
        ori_images = torch.clip(ori_images, 0, 1)

    for i in range(num_iter):
        images.requires_grad = True
        output = net(images)

        net.zero_grad()
        loss = loss_function(output, labels).cuda()
        loss.backward()

        adv_images = images + alpha * images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()

    return images


def attack_test_data(raw_test_data, net, batch_size, num_iter,eps,alpha):
    net.eval()
    c, h, w = raw_test_data[0][0].shape
    raw_train_input = []
    raw_train_label = []
    cnt = 0
    for image, label in raw_test_data:
        cnt += 1
        raw_train_input.append(image.reshape(1, c, h, w))
        raw_train_label.append(label)
    raw_train_input = torch.cat(raw_train_input, dim=0)
    raw_train_label = raw_train_label

    adversarial_train_input = []
    for i in range(0, len(raw_test_data), batch_size):
        images = raw_train_input[i:min(i + batch_size, len(raw_test_data))]
        labels = raw_train_label[i:min(i + batch_size, len(raw_test_data))]
        adversarial_batch_input = attack_single_batch_input(net, images, labels, num_iter, eps,alpha)
        adversarial_train_input.append(adversarial_batch_input)
     
    adversarial_train_input = torch.cat(adversarial_train_input, dim=0)
   
    adversarial_dataset = AdversarialDataset(adversarial_train_input, raw_train_label)
    #print(adversarial_train_input.size(),np.array(raw_train_label).shape)
    #print("Constructing adversarial dataset successfully!")

    return adversarial_dataset


def pseudo_single_batch_input(input, model, noise,possibility):
    labeles = []
    posses = []
    indexes = []
    with torch.no_grad():
        input = input.cuda()
        input_var = Variable(input)
        output = model(input_var, noise=noise)
        pred_score = F.softmax(output.data, dim=1)
        pred_score = pred_score.topk(1, dim=1, largest=True)
        poss = pred_score.values.view(-1)
        label = pred_score.indices.view(-1)
        p = torch.full_like(poss, possibility)
        index_p = torch.ge(poss, p)
        labeles.extend(label.tolist())
        posses.extend(poss.tolist())
        indexes.extend(index_p.tolist())
    return labeles, posses, indexes


def pseudo_data(unlabel_loader,net,noise,possibility):
    net.eval()

    #print("\nStart generating pseudo data...")
    pseudo_train_input = []
    pseudo_train_target = []
   
    for i, (input_, _) in enumerate(unlabel_loader):
        labeles, posses, indexes = pseudo_single_batch_input(input_, model=net,noise=noise,possibility=possibility)
   
        pseudo_batch_input=input_[indexes]
        pseudo_train_target.extend(torch.tensor(labeles)[indexes].tolist())
        pseudo_train_input.append(pseudo_batch_input)
        #pseudo_train_target.append(pseudo_batch_target)
    pseudo_train_input = torch.cat(pseudo_train_input, dim=0)
    #pseudo_train_target=torch.cat(pseudo_train_target,dim=0)
    pseudo_dataset = Pseudo_Dataset(pseudo_train_input, pseudo_train_target)
    #print(pseudo_train_input.size(),np.array(pseudo_train_target).shape)
    #print("Constructing pseudo dataset successfully!\n")
    return pseudo_dataset


def load_sub_trainset(train_data,num_classes, batch_size, workers, labels_per_class,labels_per_class_2=None):
    def get_sampler(labels, labels_per_class,labels_per_class_2=labels_per_class_2):
        if labels_per_class_2 is None:
            labels_per_class_2=len(train_data)-labels_per_class
        # Only choose digits in num_classes
        (indices,) = np.where(reduce(__or__, [labels == i for i in np.arange(num_classes)]))
        # Ensure uniform distribution of labels
        np.random.shuffle(indices)
        indices_label = np.hstack(
            [list(filter(lambda idx: labels[idx] == i, indices))[:labels_per_class] for i in range(num_classes)])
        indices_unlabel = np.hstack(
            [list(filter(lambda idx: labels[idx] == i, indices))[labels_per_class:labels_per_class+labels_per_class_2] for i in range(num_classes)])
        indices_train = np.hstack(
            [list(filter(lambda idx: labels[idx] == i, indices))[:labels_per_class+labels_per_class_2] for i in range(num_classes)])
        indices_label = torch.from_numpy(indices_label)
        indices_unlabel = torch.from_numpy(indices_unlabel)
        indices_train=torch.from_numpy(indices_train)
        sampler_label = SubsetRandomSampler(indices_label)
        sampler_unlabel = SubsetRandomSampler(indices_unlabel)
        sampler_train = SubsetRandomSampler(indices_train)
        return sampler_label, sampler_unlabel,sampler_train

    train_sampler_label, train_sampler_unlabel,train_sampler= get_sampler(train_data.targets, labels_per_class,labels_per_class_2)

    train_loader_label = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                                   sampler=train_sampler_label, shuffle=False,
                                                   num_workers=workers, pin_memory=True)

    train_loader_unlabel = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                                     sampler=train_sampler_unlabel, shuffle=False,
                                                     num_workers=workers, pin_memory=True)
    train_loader_all = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                                       sampler=train_sampler, shuffle=False,
                                                       num_workers=workers, pin_memory=True)

    return train_loader_label,train_loader_unlabel,train_loader_all



