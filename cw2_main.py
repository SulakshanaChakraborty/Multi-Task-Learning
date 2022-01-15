import load_data
import model_utils
import train_model
import test_model
import lab_loader
import train_color
import train_denoising
import denoising_loader
import data_loader_canny
import argparse


def run_cw2(args, train=True, test=False, visualize=False):
    """A function used to initiate the running of the training/testing of the model
    """
    ###############################
    # Load data
    ###############################
    train_path = 'data/train/'
    validation_path = 'data/val/'
    test_path = 'data/test/'
    batch_size = int(args.batch_size)
    device = args.device
    model_type = args.model_type
    epochs = int(args.epochs)

    print("--------------------------------------------------------------------------------")
    print(f"Model chosen: {model_type} , device: {device}, mini-batch size: {batch_size}")
    print(f"Train: {train}, Number of Epochs: {epochs}, Test: {test}")
    print("--------------------------------------------------------------------------------")

    # data loaders
    if model_type == 'MTL-segnet-with-colorization' or model_type == 'MTL-Attention-with-colorization':
        print("in here!")
        train_loader, validation_loader, test_loader = lab_loader.create_data_loaders(train_path=train_path,
                                                                                      validation_path=validation_path,
                                                                                      test_path=test_path,
                                                                                      batch_size=batch_size,
                                                                                      )

    elif model_type == 'MTL-Attention-with-denoising':
        train_loader, validation_loader, test_loader = denoising_loader.create_data_loaders(train_path=train_path,
                                                                                            validation_path=validation_path,
                                                                                            test_path=test_path,
                                                                                            batch_size=batch_size,
                                                                                            noisy=True
                                                                                            )
    elif model_type == 'MTL-Attention-with-canny':
        train_loader, validation_loader, test_loader = data_loader_canny.create_data_loaders(train_path=train_path,
                                                                                             validation_path=validation_path,
                                                                                             test_path=test_path,
                                                                                             batch_size=batch_size,
                                                                                             opencv_filters=True,
                                                                                             )

    else:
        train_loader, validation_loader, test_loader = load_data.create_data_loaders(train_path=train_path,
                                                                                     validation_path=validation_path,
                                                                                     test_path=test_path,
                                                                                     batch_size=batch_size,
                                                                                     )

    # fetch model, loss criterion and optimizer

    model, optimizer, loss_criterion = model_utils.get_model(model_type=model_type, device=device)

    ###############################
    # Train Model
    ###############################

    if train:
        print("Training the model!")

        if model_type == 'MTL-segnet-with-colorization' or model_type == 'MTL-Attention-with-colorization':

            model = train_color.train_model(model_type=model_type, train_loader=train_loader,
                                            validation_loader=validation_loader,
                                            model=model, optimizer=optimizer, loss_criterion=loss_criterion,
                                            epochs=epochs,
                                            device=device
                                            )
        elif model_type == 'MTL-Attention-with-denoising':
            model = train_denoising.train_model(model_type=model_type, train_loader=train_loader,
                                                validation_loader=validation_loader,
                                                model=model, optimizer=optimizer, loss_criterion=loss_criterion,
                                                epochs=epochs,
                                                device=device
                                                )

        elif model_type == 'MTL-Attention-with-canny':
            model = train_canny.train_model(model_type=model_type, train_loader=train_loader,
                                            validation_loader=validation_loader,
                                            model=model, optimizer=optimizer, loss_criterion=loss_criterion,
                                            epochs=epochs,
                                            device=device
                                            )

        else:
            model = train_model.train_model(model_type=model_type, train_loader=train_loader,
                                            validation_loader=validation_loader,
                                            model=model, optimizer=optimizer, loss_criterion=loss_criterion,
                                            epochs=epochs,
                                            device=device
                                            )
    ###############################
    # Test Model
    ###############################
    if test:
        print("Testing the model!")
        model_path_list = ['models/Segnet-1task-untrained.pt', 'models/Segnet-1task.pt',
                           'models/MTL-Segnet-untrained.pt', 'models/MTL-Segnet.pt'
            , 'models/MTL-Attention.pt', 'models/MTL-Attention-with-colorization.pt',
                           'models/MTL-Attention-with-denoising.pt', 'models/MTL-Attention-with-canny.pt'
            , 'models/MTL-Attention-without-bbox.pt', 'models/MTL-Attention-without-classification.pt',
                           'models/MTL-segnet-with-canny.pt'
            , 'models/MTL-segnet-with-colorization.pt']

        model_type_list = ['Segnet-1task-untrained', 'Segnet-1task', 'MTL-Segnet-untrained', 'MTL-Segnet',
                           'MTL-Attention'
            , 'MTL-Attention-with-colorization', 'MTL-Attention-with-denoising', 'MTL-Attention-with-canny'
            , 'MTL-Attention-without-bbox', 'MTL-Attention-without-classification', 'MTL-segnet-with-canny'
            , 'MTL-segnet-with-colorization']

        params_dict = dict(zip(model_type_list, model_path_list))

        model_path = params_dict[model_type]

        if model_type == 'MTL-segnet-with-colorization' or model_type == 'MTL-Attention-with-colorization':

            model, optimizer, loss_criterion = model_utils.get_model(model_type=model_type, device=device)
            train_loader, validation_loader, test_loader = lab_loader.create_data_loaders(train_path=train_path,
                                                                                          validation_path=validation_path,
                                                                                          test_path=test_path,
                                                                                          batch_size=batch_size,
                                                                                          )

            model = model_utils.load_model(model=model, model_path=model_path, device=device)
            test_model.evaluate_color_on_data(test_loader=test_loader, model=model, device=device,
                                              loss_criterion=loss_criterion, model_name=model_path)

        elif model_type == 'MTL-Attention-with-denoising':
            train_loader, validation_loader, test_loader = denoising_loader.create_data_loaders(
                train_path=train_path,
                validation_path=validation_path,
                test_path=test_path,
                batch_size=batch_size, noisy=True
            )
            model, optimizer, loss_criterion = model_utils.get_model(model_type=model_type, device=device)
            model = model_utils.load_model(model=model, model_path=model_path, device=device)

            test_model.evaluate_denoising(test_loader=test_loader, model=model, device=device,
                                          loss_criterion=loss_criterion, model_name=model_path)

        elif model_type == 'MTL-Attention-with-canny':

            train_loader, validation_loader, test_loader = load_data.create_data_loaders(train_path=train_path,
                                                                                         validation_path=validation_path,
                                                                                         test_path=test_path,
                                                                                         batch_size=batch_size)

            model, optimizer, loss_criterion = model_utils.get_model(model_type=model_type, device=device)
            model = model_utils.load_model(model=model, model_path=model_path, device=device)
            test_model.evaluate_opencv_filters(test_loader=test_loader, model=model, device=device,
                                               loss_criterion=loss_criterion, model_name=model_path)

        else:
            train_loader, validation_loader, test_loader = load_data.create_data_loaders(train_path=train_path,
                                                                                         validation_path=validation_path,
                                                                                         test_path=test_path,
                                                                                         batch_size=batch_size)

            model, optimizer, loss_criterion = model_utils.get_model(model_type=model_type, device=device)

            model = model_utils.load_model(model=model, model_path=model_path, device=device)
            test_model.evaluate_model_on_data(test_loader=test_loader, model=model, device=device,
                                              loss_criterion=loss_criterion, model_name=model_path)

    print('Completed!')


def process_args():
    """A function used to customise the running of the script.
      Various options are available such as the model type, device, batch size, mode of running (ie: training or testing).


      Returns:
          Interprets the arguments added to the argument parser object during the running of the script.
    """
    ap = argparse.ArgumentParser(description="COMP0090 cw 2 script")
    ap.add_argument("-m", '--model_type',
                    help='type of model to build (model name)',
                    default='MTL-Attention')
    ap.add_argument("-d", '--device', help='which device to run on (cuda/gpu)', default='cuda')
    ap.add_argument("-b", '--batch_size', help='mini-batch size', default=5)
    ap.add_argument("-tr", '--train', help='train the model (y/n)', default='n')
    ap.add_argument("-ts", '--test', help='test the model (y/n)', default='y')
    ap.add_argument("-e", '--epochs', help='Number of epochs', default=30)
    # ap.add_argument("-v", '--visualize', help='visualise the dataset (y/n)', default='n')
    return ap.parse_args()


if __name__ == '__main__':
    args = process_args()

    train, test = (False, False)
    if args.train == 'y': train = True
    if args.test == 'y': test = True
    # if args.visualize == 'y':   visualize = True

    run_cw2(args, train=train, test=test)  # , visualize=visualize)
