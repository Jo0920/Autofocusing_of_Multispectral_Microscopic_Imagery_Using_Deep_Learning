import numpy as np
import random

import torch.optim.optimizer
from load_data import *
from loss import *
from unet_bayesian_models import *
from unet_deterministic_models import *
from utils import *
import time


class MS_cycle_GAN_10_y:
    def __init__(self, in_channels=6, criterion_type="SSIM"):
        super(MS_cycle_GAN_10_y, self).__init__()

        # version setting
        self.time_version = time.strftime("%m-%d", time.localtime())
        self.criterion_type = criterion_type
        train_data_path = cfg.train_image_path_cycle
        # dataloader
        self.dataloader_A = DataLoader(
            MS_dataset(train_data_path, mode="only_a"),
            batch_size=1,
            shuffle=True,
            num_workers=cfg.n_cpu,
        )
        self.dataloader_B = DataLoader(
            load_simple(train_data_path + '/b'),
            batch_size=1,
            shuffle=True,
            num_workers=cfg.n_cpu,
        )
        # tensor type
        self.Tensor = torch.cuda.FloatTensor

        # generator & discriminator
        # Calculate output of image discriminator (PatchGAN)
        # self.patch = (1, cfg.patch_height // 2 ** 4, cfg.patch_width // 2 ** 4)
        self.patch = (1, 30, 30)
        # 1.Generator
        self.generator_X = GeneratorUNet_2d(in_channels=in_channels, out_channels=in_channels)
        self.generator_X = self.generator_X.cuda()
        # 2.Discriminator
        self.discriminator_X = Discriminator_2d_cycle(in_channels=in_channels)
        self.discriminator_X = self.discriminator_X.cuda()

        # init
        self.generator_X.apply(weights_init_normal)
        self.discriminator_X.apply(weights_init_normal)

        self.generator_Y = []
        self.discriminator_Y = []
        for i in range(10):
            self.generator_Y.append((GeneratorUNet_2d(in_channels=in_channels, out_channels=in_channels)).cuda())
            self.discriminator_Y.append((Discriminator_2d_cycle(in_channels=in_channels)).cuda())
            self.generator_Y[i].apply(weights_init_normal)
            self.discriminator_Y[i].apply(weights_init_normal)
        # loss
        self.criterion_GAN = torch.nn.MSELoss()
        self.criterion_GAN.cuda()

        self.criterion_ms_ssim = MS_SSIM_L1_LOSS(channel_num=in_channels)
        self.criterion_ms_ssim.cuda()

        self.criterion_l1 = torch.nn.L1Loss()
        self.criterion_l1.cuda()

        # init optimizer
        self.optimizer_G_X = torch.optim.AdamW(self.generator_X.parameters(), lr=cfg.lr_g_x,
                                               betas=(cfg.b1, cfg.b2), weight_decay=cfg.w_d)
        self.optimizer_D_X = torch.optim.AdamW(self.discriminator_X.parameters(), lr=cfg.lr_d_x,
                                               betas=(cfg.b1, cfg.b2), weight_decay=cfg.w_d)
        self.optimizer_G_Y = []
        self.optimizer_D_Y = []
        for i in range(10):
            self.optimizer_G_Y.append(
                torch.optim.AdamW(self.generator_Y[i].parameters(), lr=cfg.lr_g_y, betas=(cfg.b1, cfg.b2),
                                  weight_decay=cfg.w_d))
            self.optimizer_D_Y.append(torch.optim.AdamW(self.discriminator_Y[i].parameters(), lr=cfg.lr_d_y,
                                                        betas=(cfg.b1, cfg.b2), weight_decay=cfg.w_d))
        # images
        self.fake_B = None
        self.fake_A = None

    def train_discriminator(self, a, b, y_num):
        """
        a: NOT IN focus, tensor 6*256*256
        b:   IN   focus, tensor 6*256*256
        """
        real_A = a.unsqueeze(0)
        real_B = b.unsqueeze(0)
        real_A = Variable(real_A.type(self.Tensor))
        real_B = Variable(real_B.type(self.Tensor))

        # Adversarial ground truths
        valid = Variable(self.Tensor(np.ones((real_A.size(0), *self.patch))), requires_grad=False)
        fake = Variable(self.Tensor(np.zeros((real_A.size(0), *self.patch))), requires_grad=False)

        # ---------------------
        #  Train Discriminator
        # ---------------------
        # discriminator X checks if an image is a real in focus one
        self.fake_B = self.generator_X(real_A)
        # Real loss
        pred_real_X = self.discriminator_X(real_B)
        loss_real_X = self.criterion_GAN(pred_real_X, valid)

        # Fake loss
        pred_fake_X = self.discriminator_X(self.fake_B.detach())
        loss_fake_X = self.criterion_GAN(pred_fake_X, fake)

        # Total loss
        loss_D_X = 0.5 * (loss_real_X + loss_fake_X)

        loss_D_X.backward()

        self.fake_A = self.generator_Y[y_num](real_B)
        # Real loss
        pred_real_Y = self.discriminator_Y[y_num](real_A)
        loss_real_Y = self.criterion_GAN(pred_real_Y, valid)

        # Fake loss
        pred_fake_Y = self.discriminator_Y[y_num](self.fake_A.detach())
        loss_fake_Y = self.criterion_GAN(pred_fake_Y, fake)

        # Total loss
        loss_D_Y = 0.5 * (loss_real_Y + loss_fake_Y)

        loss_D_Y.backward()

        return loss_D_X.item(), loss_D_Y.item()

    def train_generator(self, a, b, y_num):

        real_A = a.unsqueeze(0)
        real_B = b.unsqueeze(0)
        real_A = Variable(real_A.type(self.Tensor))
        real_B = Variable(real_B.type(self.Tensor))

        # Adversarial ground truths
        valid = Variable(self.Tensor(np.ones((real_A.size(0), *self.patch))), requires_grad=False)
        fake = Variable(self.Tensor(np.zeros((real_A.size(0), *self.patch))), requires_grad=False)
        # ------------------
        #  Train Generators
        # ------------------
        # GAN loss for D_X
        # check if an image is really in focus
        pred_fake_X = self.discriminator_X(self.fake_B)
        loss_GAN_X = self.criterion_GAN(pred_fake_X, valid)

        # GAN loss for D_Y
        pred_fake_Y = self.discriminator_Y[y_num](self.fake_A)
        loss_GAN_Y = self.criterion_GAN(pred_fake_Y, valid)

        # rec_images
        rec_B = self.generator_X(self.fake_A)
        rec_A = self.generator_Y[y_num](self.fake_B)
        if self.criterion_type == "SSIM":
            # cycle loss
            # real a-> generator x -> fake b -> generator y-> rec a
            loss_cyc_a = self.criterion_ms_ssim(rec_A, real_A)
            # real b-> generator y -> fake a -> generator x -> rec b
            loss_cyc_b = self.criterion_ms_ssim(rec_B, real_B)

            # identity loss
            # real b -> generator x -> idt b
            idt_B = self.generator_X(real_B)
            loss_idt_X = self.criterion_ms_ssim(idt_B, real_B)
            # real a -> generator y -> idt a
            idt_A = self.generator_Y[y_num](real_A)
            loss_idt_Y = self.criterion_ms_ssim(idt_A, real_A)

            # Total loss
            loss_GAN = 50 * (2 * loss_GAN_X + loss_GAN_Y)
            loss_cyc = loss_cyc_a + loss_cyc_b
            loss_idt = 0.2 * (loss_idt_X + loss_idt_Y)
            loss_G = loss_GAN + loss_cyc + loss_idt

            loss_G.backward()

            return (loss_G.item(),
                    loss_cyc.item(),
                    loss_idt.item(),
                    100 * loss_GAN_X.item(),
                    50 * loss_GAN_Y.item())

        else:
            # cycle loss
            # real a-> generator x -> fake b -> generator y-> rec a
            loss_cyc_a = self.criterion_l1(rec_A, real_A)
            # real b-> generator y -> fake a -> generator x -> rec b
            loss_cyc_b = self.criterion_l1(rec_B, real_B)

            # identity loss
            # real b -> generator x -> idt b
            idt_B = self.generator_X(real_B)
            loss_idt_X = self.criterion_l1(idt_B, real_B)
            # real a -> generator y -> idt a
            idt_A = self.generator_Y[y_num](real_A)
            loss_idt_Y = self.criterion_l1(idt_A, real_A)

            # Total loss
            loss_GAN = 2 * loss_GAN_X + loss_GAN_Y
            loss_cyc = 10 * (loss_cyc_a + loss_cyc_b)
            loss_idt = 0.1 * (loss_idt_X + loss_idt_Y)
            loss_G = loss_GAN + loss_cyc + loss_idt

            loss_G.backward()

            return (loss_G.item(),
                    loss_cyc.item(),
                    loss_idt.item(),
                    2 * loss_GAN_X.item(),
                    loss_GAN_Y.item())

    def save_x(self, epoch, save_path):
        # Save model checkpoints
        if cfg.checkpoint_interval != -1 and epoch % cfg.checkpoint_interval == 0:
            torch.save(self.generator_X.state_dict(), save_path + "/generator_X_cycle_%d.pth" % (
                epoch))
            torch.save(self.discriminator_X.state_dict(), save_path + "/discriminator_X_cycle_%d.pth" % (
                epoch))
            print()
            print('x models saved!')
        else:
            print()
            print('warning! model not saved here!')

    def save_y(self, save_path):
        for i in range(10):
            torch.save(self.generator_Y[i].state_dict(), save_path + "/generator_Y_cycle_%d.pth" % (
                    i + 1))
            torch.save(self.discriminator_Y[i].state_dict(), save_path + "/discriminator_Y_cycle_%d.pth" % (
                    i + 1))
        print()
        print('y models saved!')

    def lr_decay(self, decay_rate):
        for p in self.optimizer_G_X.param_groups:
            p['lr'] *= decay_rate
        for p in self.optimizer_D_X.param_groups:
            p['lr'] *= decay_rate
        for i in range(10):
            for p in self.optimizer_G_Y[i].param_groups:
                p['lr'] *= decay_rate
            for p in self.optimizer_D_Y[i].param_groups:
                p['lr'] *= decay_rate

    def lr_reset(self, lr):
        for p in self.optimizer_G_X.param_groups:
            p['lr'] = lr
        for p in self.optimizer_D_X.param_groups:
            p['lr'] = lr
        for i in range(10):
            for p in self.optimizer_G_Y[i].param_groups:
                p['lr'] = lr
            for p in self.optimizer_D_Y[i].param_groups:
                p['lr'] = lr

    def train(self):
        cfg = get_cfg()
        save_path = "saved_models/%s_cycle_%s_%s" % (self.time_version, self.criterion_type, cfg.cat_dataset_name)
        os.makedirs(save_path, exist_ok=True)
        # Train every epoch
        for epoch in range(cfg.epoch + 1, cfg.n_epoch + 1):

            # learning rate decay
            new_lr = cfg.lr_g_x / 20 * (21 - epoch)
            self.lr_reset(new_lr)
            # train every a with a random b
            for i, a_set in enumerate(self.dataloader_A):
                G_list = []
                cyc_list = []
                idt_list = []
                ganx_list = []
                gany_list = []
                DX_list = []
                DY_list = []
                a_imgs = []
                b_imgs = []
                a_set = np.asarray(a_set).transpose()[0]
                # a infos
                image_a_name = a_set[0][-20:-12] + a_set[0][-8:-4]
                image_num = int(a_set[0][-14:-12]) - 1
                focus_num = int(a_set[0][-7:-4])
                infocus_list = np.zeros(15, dtype=np.int)
                infocus_list[0] = 13
                infocus_list[1] = 17
                infocus_list[2] = 16
                infocus_list[10] = 14
                infocus_list[12] = 14  # mark infocus list, should be modified if train data changed
                # decide to train which y
                if focus_num > infocus_list[image_num]:
                    y_num = 5 + (focus_num - infocus_list[image_num] - 1) // 2
                else:
                    y_num = 5 - (infocus_list[image_num] - focus_num + 1) // 2

                # load a images
                for a in range(cfg.spectrum_num):
                    img_tmp = Image.open(a_set[a])
                    a_imgs.append(img_tmp)
                real_a = ms_img2patch_2d(a_imgs, ispredict=False)

                # load one set b
                b_set = next(iter(self.dataloader_B))
                b_set = np.asarray(b_set).transpose()[0]
                # b infos
                image_b_name = b_set[0][-16:-8]

                # load b images
                for b in range(cfg.spectrum_num):
                    img_tmp = Image.open(b_set[b])
                    b_imgs.append(img_tmp)
                real_b = ms_img2patch_2d(b_imgs, ispredict=False)

                # train batch
                for sub in range(0, len(real_a), cfg.batch_size):
                    # train discriminator
                    # reset d
                    self.optimizer_D_X.zero_grad()
                    self.optimizer_D_Y[y_num].zero_grad()
                    for num in range(sub, min(sub + cfg.batch_size, len(real_a))):
                        lossdx, lossdy = self.train_discriminator(real_a[num], real_b[num], y_num)
                        DX_list.append(lossdx)
                        DY_list.append(lossdy)
                    # update d
                    # only update about 50 % optimizer_D_X
                    if random.randint(0, 1) == 1:
                        self.optimizer_D_X.step()
                        self.optimizer_D_Y[y_num].step()

                    # train generator
                    # reset g
                    self.optimizer_G_X.zero_grad()
                    self.optimizer_G_Y[y_num].zero_grad()
                    for num in range(sub, min(sub + cfg.batch_size, len(real_a))):
                        lossg, losscyc, lossidt, lossganx, lossgany = self.train_generator(real_a[num], real_b[num],
                                                                                           y_num)
                        G_list.append(lossg)
                        cyc_list.append(losscyc)
                        idt_list.append(lossidt)
                        ganx_list.append(lossganx)
                        gany_list.append(lossgany)
                    # update g
                    self.optimizer_G_X.step()
                    self.optimizer_G_Y[y_num].step()
                    sub += cfg.batch_size

                # compute mean
                avg_G = np.mean(G_list)
                avg_cyc = np.mean(cyc_list)
                avg_idt = np.mean(idt_list)
                avg_ganx = np.mean(ganx_list)
                avg_gany = np.mean(gany_list)
                avg_DX = np.mean(DX_list)
                avg_DY = np.mean(DY_list)
                sys.stdout.write(
                    'Train Epoch: %d, set_iter: %d, a_name: %s, b_name: %s, Y_num: %d, loss_G: %f, '
                    'loss_cyc: %f, loss_idt: %f, loss_gan_x: %f, loss_gan_y: %f, '
                    'loss_D_X: %f, loss_D_Y: %f \n' % (
                        epoch, i + 1, image_a_name, image_b_name, y_num, avg_G,
                        avg_cyc, avg_idt, avg_ganx, avg_gany,
                        avg_DX, avg_DY))
            self.save_x(epoch, save_path=save_path)
            self.save_y(save_path=save_path)
