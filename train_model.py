import sys

import torch
import numpy as np
import time


def train_model(model_type, train_loader, validation_loader, model, optimizer, loss_criterion, epochs, device):
    # todo: update with new requirements of mlt.
    # Iterate over the whole data.
    for epoch in range(epochs):

        time_epoch = time.time()
        train_loss = []
        train_accuracy = []
        for i, batch_data in enumerate(train_loader, 1):
            inputs, labels = batch_data

            inputs = inputs.to(device)
            mask = torch.squeeze(labels['mask'].to(device))
            mask = mask.to(torch.long)
            binary = torch.squeeze(labels['classification'].to(device))
            binary = binary.to(torch.long)
            bbox = labels['bbox'].to(device)

            print(binary.size(), "class shape")

            optimizer.zero_grad()
            classes, boxes, segmask = model(inputs)

            print(classes.size(), "class shape")

            # todo: update the loss_criterion in the loss computation.
            loss_seg = cri_seg(segmask, mask)
            loss_class = cri_class(classes, classes)
            loss = loss_seg + loss_class
            # todo: make the weight of losses a hyper-parameter

            # pred_ax=np.argmax(classes.detach().numpy(),axis=1)
            # train_accuracy.append(np.sum((classes.detach().numpy()==pred_ax).astype(int))/len(binary))
            train_loss.append(loss.item())

            loss.backward()
            optimizer.step()

            # todo: include validation loader in training loop (metrics)

        time_epoch_vl = time.time()
        print('----------------------------------------------------------------------------------')
        print(f"Epoch: {epoch + 1} Time taken : {round(time_epoch_vl - time_epoch, 3)} seconds")
        print("-----------------------Training Metrics-------------------------------------------")
        print("Loss: ", round(np.mean(train_loss), 3))
