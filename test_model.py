from sklearn.metrics import jaccard_score, f1_score
import torch
import numpy as np
from metrics import eval_metrics


def evaluate_model_on_data(test_loader, model, device, loss_criterion, model_name=""):
    """A function for the testing the model on the specified testing data.

    Args:
        test_loader (pytorch object): pytorch data loader for the testing set.
        model (pytorch object): pytorch model for the network.
        device (string): the device used for training of the model (cpu or cuda).
        loss_criterion (pytorch object): a loss function for the respective model type.
        model_name (str, optional): Name assigned to the model. Defaults to "".
    """
    # todo: implement test scheme.
    test_loss = []
    test_accuracy = []
    test_iou = []
    test_bbox_loss = []
    test_segmentation_loss = []
    test_label_loss = []
    test_jaca = []
    test_f1_arr = []

    # Evaluate model
    # Compute Testing metrics
    for i, batch_data in enumerate(test_loader, 1):
        # TODO: recognise denoiing, colour,
        # TODO recognise the metrics for which model
        with torch.no_grad():
            inputs, labels = batch_data
            inputs = inputs.to(device)
            mask = torch.squeeze(labels['mask'].to(device))
            mask = mask.to(torch.long)
            binary = torch.squeeze(labels['classification'].to(device))
            binary = binary.to(torch.long)
            bbox = labels['bbox'].to(device)
            bbox = bbox.float()

            classes, boxes, segmask = model(inputs)

            loss, labels_loss, segmentation_loss, bboxes_loss = loss_criterion(input_labels=classes,
                                                                               input_segmentations=segmask, \
                                                                               input_bboxes=boxes, target_labels=binary,
                                                                               target_segmentations=mask,
                                                                               target_bboxes=bbox)

            pred_ax = np.argmax(classes.detach().cpu().numpy(), axis=1)
            test_accuracy.append(np.sum((binary.detach().cpu().numpy() == pred_ax).astype(int)) / len(binary))
            test_loss.append(loss.item())

            test_label_loss.append(labels_loss.data.item())
            test_segmentation_loss.append(segmentation_loss.data.item())
            target_segmentation = torch.argmax(segmask, 1)
            test_mask_array = np.array(mask.cpu()).ravel()
            test_predicted_array = np.array(target_segmentation.cpu().ravel())
            iou = (eval_metrics(mask.cpu(), target_segmentation.cpu(), 2))
            #  print(round(iou.item(),3),"iou")
            test_jac = jaccard_score(test_mask_array, test_predicted_array, average='weighted')
            val_f1 = f1_score(test_mask_array, test_predicted_array)
            test_jaca.append(test_jac)
            test_f1_arr.append(val_f1)
            test_iou.append(iou.item())
            test_bbox_loss.append(bboxes_loss.data.item())

    print("-----------------------Testing Metrics-------------------------------------------")
    file = open("output.txt", "a")
    print("Model Name: " + str(model_name), file=file)
    print("Loss: ", round(np.mean(test_loss), 3), "Test Accu: ", round(np.mean(test_accuracy), 3), file=file)
    print("IOU: ", round(np.mean(test_iou), 3), file=file)
    print("BBOX-loss: ", round(np.mean(test_bbox_loss), 3), file=file)
    print("Segmnetaiton-loss", round(np.mean(test_segmentation_loss), 3), file=file)
    print("Label-loss", round(np.mean(test_label_loss), 3), file=file)
    print("Jac", round(np.mean(test_jaca), 3), file=file)
    print("F1s", round(np.mean(test_f1_arr), 3), file=file)
    file.close()


def evaluate_color_on_data(test_loader, model, device, loss_criterion, model_name=""):
    """A function for the testing colorisation on the specified testing data.

    Args:
        test_loader (pytorch object): pytorch data loader for the testing set.
        model (pytorch object): pytorch model for the network.
        device (string): the device used for training of the model (cpu or cuda).
        loss_criterion (pytorch object): a loss function for the respective model type.
        model_name (str, optional): Name assigned to the model. Defaults to "".
    """
    # todo: implement test scheme.
    test_loss = []
    test_accuracy = []
    test_iou = []
    test_bbox_loss = []
    test_segmentation_loss = []
    test_label_loss = []
    test_jaca = []
    test_f1_arr = []
    test_ab_loss = []

    # Evaluate model
    # Compute Testing metrics
    for i, batch_data in enumerate(test_loader, 1):
        # TODO: recognise denoiing, colour,
        # TODO recognise the metrics for which model
        with torch.no_grad():
            inputs, labels = batch_data
            inputs = inputs.to(device)
            mask = torch.squeeze(labels['mask'].to(device))
            mask = mask.to(torch.long)
            binary = torch.squeeze(labels['classification'].to(device))
            binary = binary.to(torch.long)
            bbox = labels['bbox'].to(device)
            label_ab = labels['ab'].to(device)

            bbox = bbox.float()
            classes, boxes, segmask, ab = model(inputs)

            loss, labels_loss, segmentation_loss, bboxes_loss, ab_loss = loss_criterion(input_labels=classes,
                                                                                        input_segmentations=segmask, \
                                                                                        input_bboxes=boxes,
                                                                                        input_img=label_ab,
                                                                                        target_img=ab,
                                                                                        target_labels=binary,
                                                                                        target_segmentations=mask,
                                                                                        target_bboxes=bbox)

            pred_ax = np.argmax(classes.detach().cpu().numpy(), axis=1)
            test_accuracy.append(np.sum((binary.detach().cpu().numpy() == pred_ax).astype(int)) / len(binary))
            test_loss.append(loss.item())

            test_ab_loss.append(ab_loss.data.item())

            test_label_loss.append(labels_loss.data.item())
            test_segmentation_loss.append(segmentation_loss.data.item())
            target_segmentation = torch.argmax(segmask, 1)
            test_mask_array = np.array(mask.cpu()).ravel()
            test_predicted_array = np.array(target_segmentation.cpu().ravel())
            iou = (eval_metrics(mask.cpu(), target_segmentation.cpu(), 2))
            #  print(round(iou.item(),3),"iou")
            test_jac = jaccard_score(test_mask_array, test_predicted_array, average='weighted')
            val_f1 = f1_score(test_mask_array, test_predicted_array)
            test_jaca.append(test_jac)
            test_f1_arr.append(val_f1)
            test_iou.append(iou.item())
            test_bbox_loss.append(bboxes_loss.data.item())

    print("-----------------------Testing Metrics-------------------------------------------")
    file = open("output.txt", "a")
    print("Model Name: " + str(model_name), file=file)
    print("Loss: ", round(np.mean(test_loss), 3), "Test Accu: ", round(np.mean(test_accuracy), 3), file=file)
    print("IOU: ", round(np.mean(test_iou), 3), file=file)
    print("BBOX-loss: ", round(np.mean(test_bbox_loss), 3), file=file)
    print("Segmnetaiton-loss", round(np.mean(test_segmentation_loss), 3), file=file)
    print("Label-loss", round(np.mean(test_label_loss), 3), file=file)
    print("Jac", round(np.mean(test_jaca), 3), file=file)
    print("F1s", round(np.mean(test_f1_arr), 3), file=file)
    print("AB", round(np.mean(test_ab_loss), 3), file=file)
    file.close()


def evaluate_denoising(test_loader, model, device, loss_criterion, model_name=""):
    """A function for the testing denoising on the specified testing data.

    Args:
        test_loader (pytorch object): pytorch data loader for the testing set.
        model (pytorch object): pytorch model for the network.
        device (string): the device used for training of the model (cpu or cuda).
        loss_criterion (pytorch object): a loss function for the respective model type.
        model_name (str, optional): Name assigned to the model. Defaults to "".
    """
    # todo: implement test scheme.
    test_loss = []
    test_accuracy = []
    test_iou = []
    test_jaca = []
    test_f1_arr = []
    test_denoise_jac = []

    test_bbox_loss = []
    test_segmentation_loss = []
    test_label_loss = []
    test_denoise_loss = []

    # Evaluate model
    # Compute Testing metrics
    for i, batch_data in enumerate(test_loader, 1):
        # TODO: recognise denoiing, colour,
        # TODO recognise the metrics for which model
        with torch.no_grad():
            inputs, labels = batch_data
            inputs = inputs.to(device)
            mask = torch.squeeze(labels['mask'].to(device))
            mask = mask.to(torch.long)
            binary = torch.squeeze(labels['classification'].to(device))
            binary = binary.to(torch.long)
            bbox = labels['bbox'].to(device)

            try:
                denoised_target = labels['denoised'].to(device)
            except:
                denoised_target = None

            bbox = bbox.float()
            classes, boxes, segmask, denoised_pred = model(inputs)
            loss, labels_loss, segmentation_loss, bboxes_loss, denoise_loss = loss_criterion(input_labels=classes,
                                                                                             input_segmentations=segmask, \
                                                                                             input_bboxes=boxes,
                                                                                             input_denoise=denoised_pred,
                                                                                             target_labels=binary,
                                                                                             target_segmentations=mask,
                                                                                             target_bboxes=bbox,
                                                                                             target_denoise=denoised_target)

            pred_ax = np.argmax(classes.detach().cpu().numpy(), axis=1)
            test_accuracy.append(np.sum((binary.detach().cpu().numpy() == pred_ax).astype(int)) / len(binary))
            test_loss.append(loss.item())

            # print(test_accuracy[i - 1], "minibatch acc")

            test_label_loss.append(labels_loss.data.item())
            test_segmentation_loss.append(segmentation_loss.data.item())

            test_bbox_loss.append(bboxes_loss.data.item())
            target_segmentation = torch.argmax(segmask, 1)
            iou = (eval_metrics(mask.cpu(), target_segmentation.cpu(), 2))
            test_iou.append(iou.item())
            test_denoise_loss.append(denoise_loss.data.item())

            mask_array = np.array(mask.cpu()).ravel()
            predicted_array = np.array(target_segmentation.cpu()).ravel()

            # print(jaccard_score(mask_array,predicted_array,average='weighted'),'skjac segmentation')
            # print(jaccard_score(denoise_pred_array,denoised_target_array,average='weighted'),'denoising segmentation')
            # print(f1_score(mask_array,predicted_array),"skf1")

            test_jac = jaccard_score(mask_array, predicted_array, average='weighted')
            # test_jac_denoise = jaccard_score(denoise_pred_array,denoised_target_array,average='weighted')
            test_f1 = f1_score(mask_array, predicted_array)

            test_jaca.append(test_jac)
            test_denoise_jac.append(test_jaca)
            test_f1_arr.append(test_f1)

    print("-----------------------Testing Metrics-------------------------------------------")
    file = open("output.txt", "a")
    print("Model Name: " + str(model_name), file=file)
    print("Loss: ", round(np.mean(test_loss), 3), "Test Accu: ", round(np.mean(test_accuracy), 3), file=file)
    print("IOU: ", round(np.mean(test_iou), 3), file=file)
    print("BBOX-loss: ", round(np.mean(test_bbox_loss), 3), file=file)
    print("Segmnetaiton-loss", round(np.mean(test_segmentation_loss), 3), file=file)
    print("Label-loss", round(np.mean(test_label_loss), 3), file=file)
    print("Jac", round(np.mean(test_jaca), 3), file=file)
    print("F1s", round(np.mean(test_f1_arr), 3), file=file)
    print("Denoising-loss", round(np.mean(test_denoise_loss), 3), file=file)
    file.close()
    # loss = 1
    # metrics = 2
    # return loss, metrics

    # if its segnet,


def evaluate_opencv_filters(test_loader, model, device, loss_criterion, model_name=""):
    """A function for the testing the canny filter on the specified testing data.

    Args:
        test_loader (pytorch object): pytorch data loader for the testing set.
        model (pytorch object): pytorch model for the network.
        device (string): the device used for training of the model (cpu or cuda).
        loss_criterion (pytorch object): a loss function for the respective model type.
        model_name (str, optional): Name assigned to the model. Defaults to "".
    """
    test_loss = []
    test_accuracy = []
    test_iou = []
    test_jaca = []
    test_f1_arr = []
    test_filter_jac = []

    test_bbox_loss = []
    test_segmentation_loss = []
    test_label_loss = []
    test_filter_loss = []

    # Evaluate model
    # Compute Testing metrics
    for i, batch_data in enumerate(test_loader, 1):
        with torch.no_grad():
            inputs, labels = batch_data
            inputs = inputs.to(device)
            mask = torch.squeeze(labels['mask'].to(device))
            mask = mask.to(torch.long)
            binary = torch.squeeze(labels['classification'].to(device))
            binary = binary.to(torch.long)
            bbox = labels['bbox'].to(device)
            bbox = bbox.float()

            # Add opencv filter data
            opencv_filter = labels['canny'].to(device)
            opencv_filter = opencv_filter.to(torch.long).unsqueeze(dim=1)

            # Forward pass
            classes, boxes, segmask, opencv_pred = model(inputs)

            # Loss computation
            loss, labels_loss, segmentation_loss, bboxes_loss, filters_loss = loss_criterion(input_labels=classes,
                                                                                             input_segmentations=segmask,
                                                                                             input_bboxes=boxes,
                                                                                             input_filters=opencv_pred,
                                                                                             target_labels=binary,
                                                                                             target_segmentations=mask,
                                                                                             target_bboxes=bbox,
                                                                                             target_filters=opencv_filter,
                                                                                             )

            pred_ax = np.argmax(classes.detach().cpu().numpy(), axis=1)
            test_accuracy.append(np.sum((binary.detach().cpu().numpy() == pred_ax).astype(int)) / len(binary))
            test_loss.append(loss.item())

            print(f'Mini-batch Acc: {test_accuracy[i - 1]:.4f}')

            test_label_loss.append(labels_loss.data.item())
            test_segmentation_loss.append(segmentation_loss.data.item())

            test_bbox_loss.append(bboxes_loss.data.item())
            target_segmentation = torch.argmax(segmask, 1)
            iou = (eval_metrics(mask.cpu(), target_segmentation.cpu(), 2))
            test_iou.append(iou.item())
            test_filter_loss.append(filters_loss.data.item())

            mask_array = np.array(mask.cpu()).ravel()
            predicted_array = np.array(target_segmentation.cpu()).ravel()

            test_jac = jaccard_score(mask_array, predicted_array, average='weighted')
            test_f1 = f1_score(mask_array, predicted_array)

            test_jaca.append(test_jac)
            test_filter_jac.append(test_jaca)
            test_f1_arr.append(test_f1)

    print("-----------------------Testing Metrics-------------------------------------------")
    file = open("output.txt", "a")
    print("Model Name: " + str(model_name), file=file)
    print("Loss: ", round(np.mean(test_loss), 3), "Test Accu: ", round(np.mean(test_accuracy), 3), file=file)
    print("IOU: ", round(np.mean(test_iou), 3), file=file)
    print("BBOX-loss: ", round(np.mean(test_bbox_loss), 3), file=file)
    print("Segmnetaiton-loss", round(np.mean(test_segmentation_loss), 3), file=file)
    print("Label-loss", round(np.mean(test_label_loss), 3), file=file)
    print("Jac", round(np.mean(test_jaca), 3), file=file)
    print("F1s", round(np.mean(test_f1_arr), 3), file=file)
    print("OpenCVFilter-loss", round(np.mean(test_filter_loss), 3), file=file)
    file.close()
