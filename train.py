from pix2pix import *
from cycle_gan import *


def train_model(model_name, model_type, criterion_type, conv_type, load_model=False):
    # pre-trained models will be loaded if load_model set as True
    if conv_type == "bayesian":
        model_path_g = os.path.join(cfg.cat_saved_model_path_bayesian, cfg.cat_model_name_generator)
        model_path_d = os.path.join(cfg.cat_saved_model_path_bayesian, cfg.cat_model_name_discriminator)
    else:
        model_path_g = os.path.join(cfg.cat_saved_model_path_deterministic, cfg.cat_model_name_generator)
        model_path_d = os.path.join(cfg.cat_saved_model_path_deterministic, cfg.cat_model_name_discriminator)
        
    # Bayesian convolution only provided for pix2pix
    if model_name == "pix2pix":
        if model_type == "2D":
            # simply 18 channels for 6 RGB multispectral images
            test_train = MS_GAN_pix2pix(in_channels=cfg.spectrum_num * cfg.channels,
                                        criterion_type=criterion_type,
                                        model_type=model_type,
                                        conv_type=conv_type,
                                        load_model=load_model,
                                        load_model_path_g=model_path_g,
                                        load_model_path_d=model_path_d)
            test_train.train()
        '''
        elif model_type == "3D":
            # 18 channels for 6 RGB multispectral images
            test_train = MS_GAN_pix2pix(in_channels=1,
                                        criterion_type=criterion_type,
                                        model_type=model_type,
                                        load_model=load_model,
                                        conv_type=conv_type,
                                        load_model_path_g=model_path_g,
                                        load_model_path_d=model_path_d)
            test_train.train()
        '''
        elif model_type == "2D_gray":
            # 6 channels for 6 grayscale images
            test_train = MS_GAN_pix2pix(in_channels=cfg.spectrum_num,
                                        criterion_type=criterion_type,
                                        model_type=model_type,
                                        conv_type=conv_type,
                                        load_model=load_model,
                                        load_model_path_g=model_path_g,
                                        load_model_path_d=model_path_d)
            test_train.train()
        else:
            print("Warning: No such model!")
    elif model_name == "cycle_gan":
        test_train = MS_cycle_GAN_10_y(in_channels=cfg.spectrum_num, criterion_type=criterion_type)
        test_train.train()
    else:
        print("Warning: No such model!")


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    model_name = "pix2pix"
    model_type = "2D_gray"
    criterion_type = "SSIM"
    conv_type = "bayesian"
    load_model = False
    train_model(model_name=model_name, model_type=model_type,
                criterion_type=criterion_type, conv_type=conv_type,
                load_model=load_model)
