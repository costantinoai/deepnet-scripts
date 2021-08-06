# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 19:06:35 2021

@author: 45027900
"""
import os, glob, json, collections
from natsort import natsorted

import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from fastai.vision.all import *
from kornia import rgb_to_grayscale


def label_from_path(path):
    filename = os.path.splitext(os.path.basename(name))[0]
    if "b" in filename:
        label = "bike"
    elif "c" in filename:
        label = "car"
    elif "f" in filename:
        label = "female"
    elif "m" in filename:
        label = "male"
    return label


def plot_filters(t):

    # get the number of kernels
    num_kernels = t.shape[0]

    # define number of columns for subplots
    num_cols = 12
    # rows = num of kernels
    num_rows = num_kernels

    # set the figure size
    fig = plt.figure(figsize=(num_cols, num_rows))

    # looping through all the kernels
    for i in range(t.shape[0]):
        ax1 = fig.add_subplot(num_rows, num_cols, i + 1)

        # for each kernel, we convert the tensor to numpy
        npimg = np.array(t[i].numpy(), np.float32)
        # standardize the numpy image
        npimg = (npimg - np.mean(npimg)) / np.std(npimg)
        npimg = np.minimum(1, np.maximum(0, (npimg + 0.5)))
        # npimg = npimg.transpose((1, 2, 0))
        ax1.imshow(npimg)
        ax1.axis("off")
        ax1.set_title(str(i))
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])

    # plt.savefig('myimage.png', dpi=100)
    plt.tight_layout()
    plt.show()


def get_feat_vector_l1(path_img, model, preprocess):
    input_image = Image.open(path_img)

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        my_output = None

        def my_hook(module_, input_, output_):
            nonlocal my_output
            my_output = output_

        a_hook = model.features[0].register_forward_hook(my_hook)
        model(input_batch)
        a_hook.remove()
        return my_output


def get_feat_vector_l7(path_img, model, preprocess):
    input_image = Image.open(path_img)

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        my_output = None

        def my_hook(module_, input_, output_):
            nonlocal my_output
            my_output = output_

        a_hook = model.classifier.register_forward_hook(my_hook)
        model(input_batch)
        a_hook.remove()
        return my_output


## LOAD IMAGENET LABELS
imagenet_labels_path = r"C:\Users\45027900\Desktop\project\imagenet_class_index.json"

with open(imagenet_labels_path, "r") as h:
    imagenet_labels = json.load(h)

## INIT ALEXNET
net = alexnet(pretrained=True, progress=True)

preprocess = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

## DEFINE DATASET
img_dir = r"C:\Users\45027900\Desktop\model_old\data\original"
img_labels = pd.read_csv(os.path.join(img_dir, "labels", "labels.csv"), header=None)

total_imgs = natsorted(glob.glob(os.path.join(img_dir, "*.png")))

## GET LAYERS 1 AND 7 ACTIVATIONS FOR EACH IMAGE AND STORE RESULTS IN DICT
data_dict = {
    os.path.splitext(os.path.basename(path))[0]: {
        "imgpath": path,
        "label": label_from_path(path),
        "predicted": str,
        "l1": Tensor(),
        "l7": Tensor(),
    }
    for path in total_imgs
}

for item in data_dict.keys():
    ## GET AN IMAGE
    img_path = data_dict[item]["imgpath"]
    image = Image.open(img_path)
    # plt.imshow(image)

    ## GET LAYERS 1 AND 7
    data_dict[item]["l1"] = get_feat_vector_l1(img_path, net, preprocess)
    data_dict[item]["l7"] = get_feat_vector_l7(img_path, net, preprocess)

    ## GET PREDICTED LABEL
    _, predicted = torch.max(l7.data, 1)
    data_dict[item]["predicted"] = imagenet_labels[str(int(predicted[0]))][1]

    ## PLOT PREPROCESSED IMAGE
    # tensor_img = preprocess(image).unsqueeze(0)
    # list(tensor_img.shape)
    # plt.imshow(np.transpose(tensor_img[0].cpu(), (1,2,0)))

    ## PLOT FIRST CONV LAYER
    # l1_filters = l1.squeeze(0)
    # plot_filters(l1_filters.data)

    ## PLOT PREDICTED LABEL
    # figure = plt.figure(figsize=(8, 8))
    # plt.title(data_dict[item]['predicted'])
    # plt.axis("off")
    # plt.imshow(preprocess(image).unsqueeze(0)[0].cpu().permute(1, 2, 0))
    # plt.show()
