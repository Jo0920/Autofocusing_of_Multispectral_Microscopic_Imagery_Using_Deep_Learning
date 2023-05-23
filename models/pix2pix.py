import torch.optim.optimizer
from load_data import *
from loss import *
from unet_bayesian_models import GeneratorUNet_2d_bayesian
from unet_deterministic_models import *
from utils import *
import time


class MS_GAN_pix2pix:
    def __init__(self, in_channels, model_type="2D", criterion_type="SSIM",
                 conv_type="deterministic",
                 load_model=False, load_model_path_g=None, load_model_path_d=None):
        super(MS_GAN_pix2pix, self).__init__()

        # version setting
        self.time_version = time.strftime("%m-%d", time.localtime())
        self.model_type = model_type
        self.criterion_type = criterion_type
        self.conv_type = conv_type
        self.load_model = load_model
        if self.load_model:
            self.load_model_path_d = load_model_path_d
            self.load_model_path_g = load_model_path_g
        if model_type == "2D_gray":
            train_data_path = cfg.train_image_path_gray
        else:
            train_data_path = cfg.train_image_path

        # dataloader
        self.dataloader = DataLoader(
            MS_dataset(train_data_path, mode="all"),
            batch_size=cfg.img_num,
            shuffle=True,
            num_workers=cfg.n_cpu,
        )

        # tensor type
        self.Tensor = torch.cuda.FloatTensor

        # to init
        # generator & discriminator
        self.generator = None
        self.discriminator = None
        # patch
        self.patch = None
        # loss
        self.criterion_GAN = torch.nn.MSELoss()
        self.criterion_GAN.cuda()
        if criterion_type == "SSIM":
            self.criterion_pixel = MS_SSIM_L1_LOSS(channel_num=in_channels)
        else:
            self.criterion_pixel = torch.nn.L1Loss()
        self.criterion_pixel.cuda()

        # init patch, generator and discriminator based on model type
        if not self.conv_type == "bayesian":
            if self.model_type == "2D" or "2D_gray":
                self.unet_2d_init(in_channels=in_channels)
            elif self.model_type == "3D":
                self.unet_3d_init(in_channels=in_channels)
        else:
            self.unet_2d_bayesian_init(in_channels=in_channels)

        self.generator = self.generator.cuda()
        self.discriminator = self.discriminator.cuda()

        # load preset weights if training is based on existing model
        if self.load_model:
            self.generator.load_state_dict(torch.load(self.load_model_path_g))
            self.discriminator.load_state_dict(torch.load(self.load_model_path_d))
            print("model loaded!\n")
        elif not self.conv_type == "bayesian":
            self.generator.apply(weights_init_normal)
            self.discriminator.apply(weights_init_normal)

        # init optimizer
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=cfg.lr_g, betas=(cfg.b1, cfg.b2))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=cfg.lr_d, betas=(cfg.b1, cfg.b2))

    def unet_2d_init(self, in_channels):
        # Calculate output of image discriminator (PatchGAN)
        # self.patch = (1, cfg.patch_height // 2 ** 4, cfg.patch_width // 2 ** 4)
        self.patch = (1, 30, 30)
        # 1.Generator
        self.generator = GeneratorUNet_2d(in_channels=in_channels, out_channels=in_channels)
        # 2.Discriminator
        self.discriminator = Discriminator_2d(in_channels=in_channels)

    def unet_2d_bayesian_init(self, in_channels):
        # Calculate output of image discriminator (PatchGAN)
        # self.patch = (1, cfg.patch_height // 2 ** 4, cfg.patch_width // 2 ** 4)
        self.patch = (1, 30, 30)
        # 1.Generator
        self.generator = GeneratorUNet_2d_bayesian(in_channels=in_channels, out_channels=in_channels)
        # 2.Discriminator
        self.discriminator = Discriminator_2d(in_channels=in_channels)

    def unet_3d_init(self, in_channels):
        # Calculate output of image discriminator (PatchGAN)
        self.patch = (1, cfg.spectrum_num, cfg.patch_height // 2 ** 4, cfg.patch_width // 2 ** 4)
        # 1.Generator
        self.generator = GeneratorUNet_3d(in_channels=in_channels, out_channels=in_channels)
        # 2.Discriminator
        self.discriminator = Discriminator_3d(in_channels=in_channels)

    def sample_train(self, a, b):

        """
        a: NOT IN focus
        b:   IN   focus
        """

        real_A = a.unsqueeze(0)
        real_B = b.unsqueeze(0)
        real_A = Variable(real_A.type(self.Tensor))
        real_B = Variable(real_B.type(self.Tensor))

        # Adversarial ground truths
        valid = Variable(self.Tensor(np.ones((real_A.size(0), *self.patch))), requires_grad=False)
        fake = Variable(self.Tensor(np.zeros((real_A.size(0), *self.patch))), requires_grad=False)

        complexity_cost_weight = 1
        lambda_kl = 1/1000000
        sample_nbr = 3
        loss_G = 0
        loss_D = 0

        for _ in range(sample_nbr):
            fake_B = self.generator(real_A)
            # ---------------------
            #  Train Discriminator
            # ---------------------
            # Real loss
            pred_real = self.discriminator(real_B, real_A)
            loss_real = self.criterion_GAN(pred_real, valid)

            # Fake loss
            pred_fake = self.discriminator(fake_B.detach(), real_A)
            loss_fake = self.criterion_GAN(pred_fake, fake)

            # Total loss
            loss_D += 0.5 * (loss_real + loss_fake)

            # ------------------
            #  Train Generators
            # ------------------

            # GAN loss
            pred_fake = self.discriminator(fake_B, real_A)

            loss_GAN = self.criterion_GAN(pred_fake, valid)

            # Pixel-wise loss
            loss_pixel = self.criterion_pixel(fake_B, real_B)
            # Total loss
            loss_G += cfg.lambda_pixel * loss_pixel

            loss_G += loss_GAN

            loss_G += lambda_kl * self.generator.nn_kl_divergence() * complexity_cost_weight

        loss_D /= sample_nbr
        loss_G /= sample_nbr
        loss_D.backward()
        loss_G.backward()

        return loss_G.item(), loss_D.item()

    def single_train(self, a, b):

        """
        a: NOT IN focus
        b:   IN   focus
        """

        real_A = a.unsqueeze(0)
        real_B = b.unsqueeze(0)
        real_A = Variable(real_A.type(self.Tensor))
        real_B = Variable(real_B.type(self.Tensor))

        # Adversarial ground truths
        valid = Variable(self.Tensor(np.ones((real_A.size(0), *self.patch))), requires_grad=False)
        fake = Variable(self.Tensor(np.zeros((real_A.size(0), *self.patch))), requires_grad=False)

        fake_B = self.generator(real_A)

        # ---------------------
        #  Train Discriminator
        # ---------------------
        # Real loss
        pred_real = self.discriminator(real_B, real_A)
        loss_real = self.criterion_GAN(pred_real, valid)

        # Fake loss
        pred_fake = self.discriminator(fake_B.detach(), real_A)
        loss_fake = self.criterion_GAN(pred_fake, fake)

        # Total loss
        loss_D = 0.5 * (loss_real + loss_fake)

        # ------------------
        #  Train Generators
        # ------------------

        # GAN loss
        pred_fake = self.discriminator(fake_B, real_A)
        loss_GAN = self.criterion_GAN(pred_fake, valid)

        # Pixel-wise loss
        loss_pixel = self.criterion_pixel(fake_B, real_B)
        # Total loss
        loss_G = loss_GAN + cfg.lambda_pixel * loss_pixel

        loss_G.backward()

        loss_D.backward()

        return loss_G.item(), loss_D.item()

    def save_model(self, epoch, save_path, channel_num=0):
        # Save model checkpoints
        if cfg.checkpoint_interval != -1 and epoch % cfg.checkpoint_interval == 0:
            torch.save(self.generator.state_dict(), save_path + "/generator_%d_%d.pth" % (
                epoch, channel_num))
            torch.save(self.discriminator.state_dict(), save_path + "/discriminator_%d_%d.pth" % (
                epoch, channel_num))
            print()
            print('model saved!\n')
        else:
            print()
            print('warning! model not saved here!\n')

    def train(self):
        """
        2D
        a: NOT IN focus, tensor 18*256*256
        b:   IN   focus, tensor 18*256*256

        2D gray:
        a: NOT IN focus, tensor 6*256*256
        b:   IN   focus, tensor 6*256*256

        3D
        a: NOT IN focus, tensor 1*18*256*256
        b:   IN   focus, tensor 1*18*256*256
        """

        cfg = get_cfg()
        save_path = "saved_models/%s_pix2pix_%s_%s_%s_%s" % (
            self.time_version, self.model_type, self.conv_type, self.criterion_type, cfg.cat_dataset_name)
        os.makedirs(save_path, exist_ok=True)

        # Train every epoch
        for epoch in range(cfg.epoch + 1, cfg.n_epoch + 1):
            # Train every image set
            for i, image_set in enumerate(self.dataloader):
                image_set = np.transpose(image_set)
                G_list = []
                D_list = []
                for pair_p in image_set:
                    a_imgs = []
                    b_imgs = []
                    for a in range(cfg.spectrum_num):
                        img_tmp = Image.open(pair_p[a])
                        a_imgs.append(img_tmp)
                    for a in range(cfg.spectrum_num, cfg.spectrum_num * 2):
                        img_tmp = Image.open(pair_p[a])
                        b_imgs.append(img_tmp)

                    real_a = ms_img2patch_2d(a_imgs, ispredict=False)
                    real_b = ms_img2patch_2d(b_imgs, ispredict=False)
                    # train batch
                    for sub in range(0, len(real_a), cfg.batch_size):
                        # reset grads
                        self.optimizer_G.zero_grad()
                        self.optimizer_D.zero_grad()
                        for num in range(sub, min(sub + cfg.batch_size, len(real_a))):
                            if self.conv_type == "bayesian":
                                loss_G, loss_D = self.sample_train(real_a[num], real_b[num])
                            elif self.model_type == "2D" or "2D_gray":
                                loss_G, loss_D = self.single_train(real_a[num], real_b[num])
                            elif self.model_type == "3D":
                                loss_G, loss_D = self.single_train(real_a[num].unsqueeze(0),
                                                                   real_b[num].unsqueeze(0))
                            G_list.append(loss_G)
                            D_list.append(loss_D)
                        sub += cfg.batch_size
                        # update grads
                        self.optimizer_G.step()
                        self.optimizer_D.step()
                avg_G = sum(G_list) / len(G_list)
                avg_D = sum(D_list) / len(D_list)
                sys.stdout.write(
                    'Train Epoch: %d , Set: %d ,loss_G: %f , loss_D: %f \n' % (
                        epoch, i + 1, avg_G, avg_D))
            self.save_model(epoch, save_path=save_path)
