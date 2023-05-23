import os

import torch
from torchvision import transforms
from torchvision.utils import save_image
from torch.autograd import Variable
from unet_deterministic_models import *
from unet_bayesian_models import *
from load_data import *
from utils import *
import matplotlib.pyplot as plt
import pylab

def predict_gray():
    # set parameters

    cfg = get_cfg()

    model_path = os.path.join(cfg.cat_saved_model_path_deterministic, cfg.cat_model_name_generator)
    img_path = cfg.test_image_path_gray
    save_to_path = os.path.join(cfg.cat_save_to, os.path.basename(cfg.cat_saved_model_path_deterministic))

    # set model
    generator = GeneratorUNet_2d(in_channels=cfg.spectrum_num, out_channels=cfg.spectrum_num, droput=0.0)
    generator = generator.cuda()
    generator.load_state_dict(torch.load(model_path))

    # convert image

    data = MS_dataset(img_path, mode="only_a")

    Tensor = torch.cuda.FloatTensor

    patch = (1, cfg.patch_height // 2 ** 4, cfg.patch_width // 2 ** 4)

    for img_set_path in tqdm(data):
        # cut image
        images_ori = []
        for a in range(cfg.spectrum_num):
            images_ori.append(Image.open(img_set_path[a]).convert("L"))
        images_cut = ms_img2patch_2d(images_ori, ispredict=True)
        # feed into generator
        generated_list = []
        for img_frag in images_cut:
            # load image
            orig_img = img_frag.unsqueeze(0)
            orig_img = Variable(orig_img.type(Tensor))
            # Adversarial ground truths
            valid = Variable(Tensor(np.ones((orig_img.size(0), *patch))), requires_grad=False)
            # into generator
            with torch.no_grad():
                generated_img = generator(orig_img)
            # generated_img = orig_img
            generated_list.append(generated_img)
            # concatenate image in generated list
        result = patch2img(generated_list)
        for a in range(cfg.spectrum_num):
            final_img = result[:, a, :, :]
            # save final big image
            os.makedirs('%s/%s/%s' % (save_to_path, img_set_path[a][-20:-12], img_set_path[a][-6:-4]), exist_ok=True)
            save_image(final_img, '%s/%s/%s/%s' % (
                save_to_path, img_set_path[a][-20:-12], img_set_path[a][-6:-4], img_set_path[a][-20:]), normalize=False)
        # Caution: parameter |normalize| in func save_image() must be False.
        torch.cuda.empty_cache()
    print("done.")


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    predict_gray()
