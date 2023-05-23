import torch
from torchvision.utils import save_image
from torch.autograd import Variable
from unet_deterministic_models import *
from unet_bayesian_models import *
from load_data import *
from utils import *
from preprocess_gray import gamma_correction
import matplotlib.pyplot as plt
import pylab
import cv2

def evaluate_regression(regressor,
                        X,
                        y,
                        samples = 100,
                        std_multiplier=2):
    preds = [regressor(X) for i in range(samples)]
    preds = torch.stack(preds)
    means = preds.mean(axis=0)
    stds = preds.std(axis=0)
    ci_upper = means + (std_multiplier * stds)
    ci_lower = means - (std_multiplier * stds)
    ic_acc = (ci_lower <= y) * (ci_upper >= y)
    ic_acc = ic_acc.float().mean()
    return ic_acc, (ci_upper >= y).float().mean(), (ci_lower <= y).float().mean()

def predict_uncertainty():
    # set parameters

    cfg = get_cfg()

    model_path = os.path.join(cfg.cat_saved_model_path_bayesian, cfg.cat_model_name_generator)
    img_path = cfg.test_image_path_gray
    save_to_path_uncertainty = os.path.join("uncertainty_estimation", os.path.basename(cfg.cat_saved_model_path_bayesian))
    save_to_path_img = os.path.join("pred_res", os.path.basename(cfg.cat_saved_model_path_bayesian))
    # set model
    generator = GeneratorUNet_2d_bayesian(in_channels=cfg.spectrum_num, out_channels=cfg.spectrum_num)
    generator = generator.cuda()
    generator.load_state_dict(torch.load(model_path))

    # convert image

    data = MS_dataset(img_path, mode="only_a")

    Tensor = torch.cuda.FloatTensor

    for img_set_path in tqdm(data):
        # cut image
        images_ori = []
        for a in range(cfg.spectrum_num):
            images_ori.append(Image.open(img_set_path[a]).convert("L"))
        x = 0
        y = 0
        images_cut = ms_img2patch_2d(images_ori, ispredict=True)#, spectrum_num=1)
        # feed into generator
        generated_list = [[],[],[],[],[],[]]
        uncertainty_list = [[],[],[],[],[],[]]
        preds = list()
        for j in range(80):
            generated_list = []
            for img_frag in images_cut:
                # load image
                with torch.no_grad():
                    orig_img = img_frag.unsqueeze(0)
                    orig_img = Variable(orig_img.type(Tensor))
                    generated_list.append(generator(orig_img))
                    torch.cuda.empty_cache()
            final_img = patch2img(generated_list)
            preds.append(final_img)
        preds = torch.stack(preds)
        generated_img_mean = preds.mean(axis=0)
        stds = preds.std(axis=0)
        #std_pdf = stds.std()
        #print(torch.mean(stds) + std_pdf * 2)
        print(torch.max(stds))
        print(torch.min(stds))
        print(torch.mean(stds))
        uncertainty_map = stds
        torch.cuda.empty_cache()
        for a in range(cfg.spectrum_num):
            # concatenate image in generated list
            final_img = generated_img_mean[:, a, :, :]
            #final_img = patch2img(generated_list[a])
            # save final big image
            os.makedirs('%s/%s/%s' % (save_to_path_img, img_set_path[a][-20:-12], img_set_path[a][-6:-4]), exist_ok=True)
            save_image(final_img, '%s/%s/%s/%s' % (
                save_to_path_img, img_set_path[a][-20:-12], img_set_path[a][-6:-4], img_set_path[a][-20:]), normalize=False)
            # concatenate image in uncertainty list
            final_map = uncertainty_map[:, a, :, :]
            #final_map = patch2img(uncertainty_list[a])
            # save final big image
            os.makedirs('%s/%s/%s' % (save_to_path_uncertainty, img_set_path[a][-20:-12], img_set_path[a][-6:-4]),
                        exist_ok=True)
            save_image(final_map, '%s/%s/%s/%s' % (
                save_to_path_uncertainty, img_set_path[a][-20:-12], img_set_path[a][-6:-4], img_set_path[a][-20:]),
                       normalize=True)

        # Caution: parameter |normalize| in func save_image() must be False.
        torch.cuda.empty_cache()

    print("done.")

def gray2heat(path_load, path_save):
    filesList = os.listdir(path_load)
    for fileName in filesList:
        fileAbpath = os.path.join(path_load, fileName)
        fileSApath = os.path.join(path_save, fileName)
        if os.path.isdir(fileAbpath):
            os.makedirs(fileSApath, exist_ok=True)
            gray2heat(fileAbpath, fileSApath)
        else:
            print(fileAbpath)
            img = cv2.imread(fileAbpath, 0)
            new_img = cv2.applyColorMap(img, cv2.COLORMAP_TURBO)
            cv2.imwrite(fileSApath, new_img)

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    predict_uncertainty()

    path_origin = "uncertainty_estimation"
    path_new = "uncertainty_estimation_heatmap"
    gray2heat(path_origin, path_new)
