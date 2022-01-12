import load_data
import model_utils
import train_model
import test_model
import torch
import displaying
import lab_loader
import train_color
import train_denoising
import denoising_loader


def run_cw2(train=True, test=False, visualize=True):
    ###############################
    # Load data
    ###############################
    train_path = 'data/train/'
    validation_path = 'data/val/'
    test_path = 'data/test/'
    batch_size = 3
    device='cuda'
 
    train_loader, validation_loader, test_loader = load_data.create_data_loaders(train_path=train_path,
                                                                                 validation_path=validation_path,
                                                                                 test_path=test_path,
                                                                                 batch_size=batch_size,
                                                                                 )

    ###############################
    # Train Model
    ###############################
    model_type = 'baseline'  # baseline' or 'mlt_hard' or 'mlt_attention' or 'denoising_attention'
    model, optimizer, loss_criterion = model_utils.get_model(model_type=model_type, device=device)

   

    #model_type = 'color_segnet' #baseline' or 'mlt_hard' or 'mlt_attention' or 'mlt_gscnn'\
    if model_type == 'color_segnet':
        train_loader, validation_loader, test_loader = lab_loader.create_data_loaders(train_path=train_path,
                                                                                 validation_path=validation_path,
                                                                                 test_path=test_path,
                                                                                 batch_size=batch_size,
                                                                                 )

    
    if train and  model_type== 'color_segnet':
     print("Training the model!")
     model = train_color.train_model(model_type=model_type, train_loader=train_loader,
                                        validation_loader=validation_loader,
                                        model=model, optimizer=optimizer, loss_criterion=loss_criterion,
                                        epochs=30,
                                        device=device
                                        )
      
    if model_type == 'denoising_attention':
      train_loader, validation_loader, test_loader = denoising_loader.create_data_loaders(train_path=train_path,
                                                                              validation_path=validation_path,
                                                                              test_path=test_path,
                                                                              batch_size=batch_size,noisy=True
                                                                              )

    
    if train and  model_type== 'denoising_attention':
      print("Training the model!")
      model = train_denoising.train_model(model_type=model_type, train_loader=train_loader,
                                        validation_loader=validation_loader,
                                        model=model, optimizer=optimizer, loss_criterion=loss_criterion,
                                        epochs=30,
                                        device=device
                                        )
        
        # Train model
    
    
    elif train:
     print("Training the model!")
     model = train_model.train_model(model_type=model_type, train_loader=train_loader,
                                        validation_loader=validation_loader,
                                        model=model, optimizer=optimizer, loss_criterion=loss_criterion,
                                        epochs=30,
                                        device=device
                                        )
    ###############################
    # Test Model
    ###############################
    model_path_list_attention=[]
    model_path_canny=[]
    model_path_list_seg=[]
    model_path_list_color=[]
    model_path_list_denoising =[]
    if test:
        if model_path_canny:
         for model_path in model_path_canny:
            model_type = 'attention_opencv_filter'  
            model, optimizer, loss_criterion = model_utils.get_model(model_type=model_type, device=device)
            print("Evaluating the model!")
            model = model_utils.load_model(model=model,model_path=model_path)
            test_model.evaluate_model_on_data(test_loader=test_loader, model=model, device=device, loss_criterion=loss_criterion,model_name=model_path)
       
        if model_path_list_attention:
         for model_path in model_path_list_attention:
            train_loader, validation_loader, test_loader = load_data.create_data_loaders(train_path=train_path,
                                                                                 validation_path=validation_path,
                                                                                 test_path=test_path,
                                                                                 batch_size=batch_size,
                                                                                 )
            model_type = 'mlt_attention'  
            model, optimizer, loss_criterion = model_utils.get_model(model_type=model_type, device=device)
            print("Evaluating the model!")
            model = model_utils.load_model(model=model,model_path=model_path)
            test_model.evaluate_model_on_data(test_loader=test_loader, model=model, device=device, loss_criterion=loss_criterion,model_name=model_path)
       
        if model_path_list_seg:
         for model_path in model_path_list_seg:
          model_type = 'baseline'  
          train_loader, validation_loader, test_loader = load_data.create_data_loaders(train_path=train_path,
                                                                                 validation_path=validation_path,
                                                                                 test_path=test_path,
                                                                                 batch_size=batch_size,
                                                                                 )
          model, optimizer, loss_criterion = model_utils.get_model(model_type=model_type, device=device)
          model = model_utils.load_model(model=model,model_path=model_path)
        # Evaluate over testing dataset.
          print("Evaluating the model!")
          test_model.evaluate_model_on_data(test_loader=test_loader, model=model, device=device, loss_criterion=loss_criterion,model_name=model_path)
       
        if model_path_list_color:
         for model_path in model_path_list_color:
          model_type = 'color_segnet'  
          train_loader, validation_loader, test_loader = lab_loader.create_data_loaders(train_path=train_path,
                                                                                 validation_path=validation_path,
                                                                                 test_path=test_path,
                                                                                 batch_size=batch_size,
                                                                                 )
                                
          model, optimizer, loss_criterion = model_utils.get_model(model_type=model_type, device=device)
          model = model_utils.load_model(model=model,model_path=model_path)

        if model_path_list_denoising:
         for model_path in model_path_list_denoising:
          model_type = 'denoising_attention'  
          train_loader, validation_loader, test_loader = denoising_loader.create_data_loaders(train_path=train_path,
                                                                                 validation_path=validation_path,
                                                                                 test_path=test_path,
                                                                                 batch_size=batch_size,noisy = True
                                                                                 )
          model, optimizer, loss_criterion = model_utils.get_model(model_type=model_type, device=device)
          model = model_utils.load_model(model=model,model_path=model_path)



        # Evaluate over testing dataset.
          print("Evaluating the model!")
          test_model.evaluate_denoising(test_loader=test_loader, model=model, device=device, loss_criterion=loss_criterion,model_name=model_path)

    ###############################
    # Run visualization
    ###############################
    if visualize:
        images, labels, segmentations, bboxes = load_data.take_random_samples(data_loader=test_loader, n_samples=16)
        displaying.visualise_results(model=model, images=images, labels=labels, segmentation=segmentations,
                                     bboxes=bboxes)

    print('CW2 is done! Well, almost done.')


if __name__ == '__main__':
    run_cw2(train=False,test=True,visualize=False)
