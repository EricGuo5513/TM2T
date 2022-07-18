import torch
import random
from networks.modules import *
from networks.transformer import *
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import tensorflow as tf
from collections import OrderedDict
from utils.utils import *
from os.path import join as pjoin
from data.dataset import collate_fn
import codecs as cs


class Logger(object):
  def __init__(self, log_dir):
    self.writer = tf.summary.create_file_writer(log_dir)

  def scalar_summary(self, tag, value, step):
      with self.writer.as_default():
          tf.summary.scalar(tag, value, step=step)
          self.writer.flush()


class Trainer(object):


    @staticmethod
    def zero_grad(opt_list):
        for opt in opt_list:
            opt.zero_grad()

    @staticmethod
    def clip_norm(network_list):
        for network in network_list:
            clip_grad_norm_(network.parameters(), 0.5)

    @staticmethod
    def step(opt_list):
        for opt in opt_list:
            opt.step()

    def ones_like(self, tensor, val=1.):
        return torch.FloatTensor(tensor.size()).fill_(val).to(self.opt.gpu_id)

    # @staticmethod
    def zeros_like(self, tensor, val=0.):
        return torch.FloatTensor(tensor.size()).fill_(val).to(self.opt.gpu_id)

    def forward(self, batch_data):
        pass

    def backward(self):
        pass

    def update(self):
        pass


class VQTokenizerTrainer(Trainer):
    def __init__(self, args, vq_encoder, quantizer, vq_decoder, discriminator=None):
        self.opt = args
        self.vq_encoder = vq_encoder
        self.vq_decoder = vq_decoder
        self.quantizer = quantizer
        self.discriminator = discriminator
        self.device = args.device

        if args.is_train:
            self.logger = Logger(args.log_dir)
            self.l1_criterion = torch.nn.L1Loss()
            self.gan_criterion = torch.nn.BCEWithLogitsLoss()

    def ones_like(self, tensor, val=1.):
        return torch.FloatTensor(tensor.size()).fill_(val).to(self.opt.gpu_id)

    # @staticmethod
    def zeros_like(self, tensor, val=0.):
        return torch.FloatTensor(tensor.size()).fill_(val).to(self.opt.gpu_id)

    def forward(self, batch_data):
        motions = batch_data
        self.motions = motions.detach().to(self.device).float()
        # print(self.motions.shape)
        self.pre_latents = self.vq_encoder(self.motions[..., :-4])
        # print(self.pre_latents.shape)
        self.embedding_loss, self.vq_latents, _, self.perplexity = self.quantizer(self.pre_latents)
        # print(self.vq_latents.shape)
        self.recon_motions = self.vq_decoder(self.vq_latents)
        # print(self.recon_motions.shape)

    def backward_G(self):
        self.loss_rec = self.l1_criterion(self.recon_motions, self.motions)

        self.loss_G = self.loss_rec + self.embedding_loss

        if self.opt.use_gan:
            fake_feats, fake_labels = self.discriminator(self.recon_motions)
            ones = self.ones_like(fake_labels)
            self.loss_G_adv = self.gan_criterion(fake_labels, ones)
            self.loss_G += self.loss_G_adv * self.opt.lambda_adv

            if self.opt.use_feat_M:
                self.loss_G_FM = self.l1_criterion(fake_feats, self.real_feats.detach())
                self.loss_G += self.loss_G_FM * self.opt.lambda_fm

    def backward_D(self):
        self.real_feats, real_labels = self.discriminator(self.motions.detach())
        fake_feats, fake_labels = self.discriminator(self.recon_motions.detach())

        ones = self.ones_like(real_labels)
        zeros = self.zeros_like(fake_labels)
        self.loss_D_T = self.gan_criterion(real_labels, ones)
        self.loss_D_F = self.gan_criterion(fake_labels, zeros)
        self.loss_D = (self.loss_D_T + self.loss_D_F) * self.opt.lambda_adv


    def update(self):
        loss_logs = OrderedDict({})

        if self.opt.use_gan:
            self.zero_grad([self.opt_discriminator])
            self.backward_D()
            self.loss_D.backward(retain_graph=True)
            self.step([self.opt_discriminator])

        self.zero_grad([self.opt_vq_encoder, self.opt_quantizer, self.opt_vq_decoder])
        self.backward_G()
        self.loss_G.backward()
        self.step([self.opt_vq_encoder, self.opt_quantizer, self.opt_vq_decoder])
        loss_logs['loss_G'] = self.loss_G.item()
        loss_logs['loss_G_rec'] = self.loss_rec.item()
        loss_logs['loss_G_emb'] = self.embedding_loss.item()
        loss_logs['perplexity'] = self.perplexity.item()

        if self.opt.use_gan:
            loss_logs['loss_G_adv'] = self.loss_G_adv.item()
            if self.opt.use_feat_M:
                loss_logs['loss_G_FM'] = self.loss_G_FM.item()

            loss_logs['loss_D'] = self.loss_D.item()
            loss_logs['loss_D_T'] = self.loss_D_T.item()
            loss_logs['loss_D_F'] = self.loss_D_F.item()

        return loss_logs

    def save(self, file_name, ep, total_it):
        if self.opt.use_gan:
            state = {
                'vq_encoder': self.vq_encoder.state_dict(),
                'quantizer': self.quantizer.state_dict(),
                'vq_decoder': self.vq_decoder.state_dict(),
                'discriminator': self.discriminator.state_dict(),

                'opt_vq_encoder': self.opt_vq_encoder.state_dict(),
                'opt_quantizer': self.opt_quantizer.state_dict(),
                'opt_vq_decoder': self.opt_vq_decoder.state_dict(),
                'opt_discriminator': self.opt_discriminator.state_dict(),

                'ep': ep,
                'total_it': total_it,
            }
            # state['discriminator'] = self.discriminator.state_dict()
            # state['opt_discriminator'] = self.opt_discriminator.state_dict()
        else:
            state = {
                'vq_encoder': self.vq_encoder.state_dict(),
                'quantizer': self.quantizer.state_dict(),
                'vq_decoder': self.vq_decoder.state_dict(),

                'opt_vq_encoder': self.opt_vq_encoder.state_dict(),
                'opt_quantizer': self.opt_quantizer.state_dict(),
                'opt_vq_decoder': self.opt_vq_decoder.state_dict(),

                'ep': ep,
                'total_it': total_it,
            }
        torch.save(state, file_name)

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        self.vq_encoder.load_state_dict(checkpoint['vq_encoder'])
        self.quantizer.load_state_dict(checkpoint['quantizer'])
        self.vq_decoder.load_state_dict(checkpoint['vq_decoder'])

        self.opt_vq_encoder.load_state_dict(checkpoint['opt_vq_encoder'])
        self.opt_quantizer.load_state_dict(checkpoint['opt_quantizer'])
        self.opt_vq_decoder.load_state_dict(checkpoint['opt_vq_decoder'])

        if self.opt.use_gan:
            self.discriminator.load_state_dict(checkpoint['discriminator'])
            self.opt_discriminator.load_state_dict(checkpoint['opt_discriminator'])
        return checkpoint['ep'], checkpoint['total_it']

    def train(self, train_dataloader, val_dataloader, plot_eval):
        self.vq_encoder.to(self.device)
        self.quantizer.to(self.device)
        self.vq_decoder.to(self.device)

        self.opt_vq_encoder = optim.Adam(self.vq_encoder.parameters(), lr=self.opt.lr)
        self.opt_quantizer = optim.Adam(self.quantizer.parameters(), lr=self.opt.lr)
        self.opt_vq_decoder = optim.Adam(self.vq_decoder.parameters(), lr = self.opt.lr)

        if self.opt.use_gan:
            self.discriminator.to(self.device)
            self.opt_discriminator = optim.Adam(self.discriminator.parameters(), lr=self.opt.lr*0.1)

        epoch = 0
        it = 0
        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')
            epoch, it = self.resume(model_dir)

        start_time = time.time()
        total_iters = self.opt.max_epoch * len(train_dataloader)
        print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_dataloader), len(val_dataloader)))
        val_loss = 0
        min_val_loss = np.inf
        min_val_epoch = epoch
        logs = OrderedDict()
        while epoch < self.opt.max_epoch:
            for i, batch_data in enumerate(train_dataloader):
                self.vq_encoder.train()
                self.quantizer.train()
                self.vq_decoder.train()

                if self.opt.use_gan:
                    self.discriminator.train()

                self.forward(batch_data)

                log_dict = self.update()
                # time3 = time.time()
                # print('Update Time: %.5f s' % (time3 - time2))
                # time0 = time3
                for k, v in log_dict.items():
                    if k not in logs:
                        logs[k] = v
                    else:
                        logs[k] += v

                it += 1
                if it % self.opt.log_every == 0:
                    mean_loss = OrderedDict({'val_loss': val_loss})
                    self.logger.scalar_summary('val_loss', val_loss, it)

                    for tag, value in logs.items():
                        self.logger.scalar_summary(tag, value / self.opt.log_every, it)
                        mean_loss[tag] = value / self.opt.log_every
                    logs = OrderedDict()
                    print_current_loss(start_time, it, total_iters, mean_loss, epoch, i)

                if it % self.opt.save_latest == 0:
                    self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            epoch += 1
            if epoch % self.opt.save_every_e == 0:
                self.save(pjoin(self.opt.model_dir, 'E%04d.tar' % (epoch)), epoch, total_it=it)

            print('Validation time:')

            val_loss_rec = 0
            val_loss_emb = 0
            with torch.no_grad():
                for i, batch_data in enumerate(val_dataloader):
                    self.forward(batch_data)
                    val_loss_rec += self.l1_criterion(self.recon_motions, self.motions).item()
                    # val_loss_emb += self.embedding_loss.item()

            val_loss = val_loss_rec / (len(val_dataloader) + 1)
            # val_loss = val_loss / (len(val_dataloader) + 1)
            # val_loss_rec = val_loss_rec / (len(val_dataloader) + 1)
            # val_loss_emb = val_loss_emb / (len(val_dataloader) + 1)

            print('Validation Loss: %.5f' % (val_loss))

            if val_loss < min_val_loss:
                min_val_loss = val_loss
                min_val_epoch = epoch
                self.save(pjoin(self.opt.model_dir, 'finest.tar'), epoch, it)
                print('Best Validation Model So Far!~')

            if epoch % self.opt.eval_every_e == 0:
                data = torch.cat([self.recon_motions[:4], self.motions[:4]], dim=0).detach().cpu().numpy()
                save_dir = pjoin(self.opt.eval_dir, 'E%04d' % (epoch))
                os.makedirs(save_dir, exist_ok=True)
                plot_eval(data, save_dir)

            if epoch - min_val_epoch >= 5:
                print('Early Stopping!~')
                # break


class VQTokenizerTrainerV2(Trainer):
    def __init__(self, args, vq_encoder, quantizer, vq_decoder, mov_encoder=None, discriminator=None):
        self.opt = args
        self.vq_encoder = vq_encoder
        self.vq_decoder = vq_decoder
        self.quantizer = quantizer
        self.mov_encoder = mov_encoder
        self.discriminator = discriminator
        self.device = args.device

        if args.is_train:
            self.logger = Logger(args.log_dir)
            self.l1_criterion = torch.nn.L1Loss()
            self.gan_criterion = torch.nn.BCEWithLogitsLoss()

    def ones_like(self, tensor, val=1.):
        return torch.FloatTensor(tensor.size()).fill_(val).to(self.opt.gpu_id)

    # @staticmethod
    def zeros_like(self, tensor, val=0.):
        return torch.FloatTensor(tensor.size()).fill_(val).to(self.opt.gpu_id)

    def forward(self, batch_data):
        motions = batch_data
        self.motions = motions.detach().to(self.device).float()
        # print(self.motions.shape)
        self.pre_latents = self.vq_encoder(self.motions[..., :-4])
        # print(self.pre_latents.shape)
        self.embedding_loss, self.vq_latents, _, self.perplexity = self.quantizer(self.pre_latents)
        # print(self.vq_latents.shape)
        self.recon_motions = self.vq_decoder(self.vq_latents)
        # print(self.recon_motions.shape)

        if self.opt.use_percep:
            self.recon_mov = self.mov_encoder(self.recon_motions[..., :-4].clone())
            self.real_mov = self.mov_encoder(self.motions[..., :-4])

    def backward_G(self):
        self.loss_rec_mot = self.l1_criterion(self.recon_motions, self.motions)
        self.loss_G = self.loss_rec_mot + self.embedding_loss

        if self.opt.use_percep:
            self.loss_rec_mov = self.l1_criterion(self.recon_mov, self.real_mov.detach())
            self.loss_G += self.loss_rec_mov

        if self.opt.start_use_gan:
            fake_feats, fake_labels = self.discriminator(self.recon_motions)
            ones = self.ones_like(fake_labels)
            self.loss_G_adv = self.gan_criterion(fake_labels, ones)
            self.loss_G += self.loss_G_adv * self.opt.lambda_adv

            if self.opt.use_feat_M:
                self.loss_G_FM = self.l1_criterion(fake_feats, self.real_feats.detach())
                self.loss_G += self.loss_G_FM * self.opt.lambda_fm

    def backward_D(self):
        self.real_feats, real_labels = self.discriminator(self.motions.detach())
        fake_feats, fake_labels = self.discriminator(self.recon_motions.detach())

        ones = self.ones_like(real_labels)
        zeros = self.zeros_like(fake_labels)
        self.loss_D_T = self.gan_criterion(real_labels, ones)
        self.loss_D_F = self.gan_criterion(fake_labels, zeros)
        self.loss_D = (self.loss_D_T + self.loss_D_F) * self.opt.lambda_adv


    def update(self):
        loss_logs = OrderedDict({})

        if self.opt.start_use_gan:
            self.zero_grad([self.opt_discriminator])
            self.backward_D()
            self.loss_D.backward(retain_graph=True)
            self.step([self.opt_discriminator])

        self.zero_grad([self.opt_vq_encoder, self.opt_quantizer, self.opt_vq_decoder])
        self.backward_G()
        self.loss_G.backward()
        self.step([self.opt_vq_encoder, self.opt_quantizer, self.opt_vq_decoder])
        loss_logs['loss_G'] = self.loss_G.item()
        loss_logs['loss_G_rec_mot'] = self.loss_rec_mot.item()
        loss_logs['loss_G_emb'] = self.embedding_loss.item()
        if self.opt.use_percep:
            loss_logs['loss_G_rec_mov'] = self.loss_rec_mov.item()
        loss_logs['perplexity'] = self.perplexity.item()

        if self.opt.start_use_gan:
            loss_logs['loss_G_adv'] = self.loss_G_adv.item()
            if self.opt.use_feat_M:
                loss_logs['loss_G_FM'] = self.loss_G_FM.item()

            loss_logs['loss_D'] = self.loss_D.item()
            loss_logs['loss_D_T'] = self.loss_D_T.item()
            loss_logs['loss_D_F'] = self.loss_D_F.item()

        return loss_logs

    def save(self, file_name, ep, total_it):
        if self.opt.use_gan:
            state = {
                'vq_encoder': self.vq_encoder.state_dict(),
                'quantizer': self.quantizer.state_dict(),
                'vq_decoder': self.vq_decoder.state_dict(),
                'discriminator': self.discriminator.state_dict(),

                'opt_vq_encoder': self.opt_vq_encoder.state_dict(),
                'opt_quantizer': self.opt_quantizer.state_dict(),
                'opt_vq_decoder': self.opt_vq_decoder.state_dict(),
                'opt_discriminator': self.opt_discriminator.state_dict(),

                'ep': ep,
                'total_it': total_it,
            }
            # state['discriminator'] = self.discriminator.state_dict()
            # state['opt_discriminator'] = self.opt_discriminator.state_dict()
        else:
            state = {
                'vq_encoder': self.vq_encoder.state_dict(),
                'quantizer': self.quantizer.state_dict(),
                'vq_decoder': self.vq_decoder.state_dict(),

                'opt_vq_encoder': self.opt_vq_encoder.state_dict(),
                'opt_quantizer': self.opt_quantizer.state_dict(),
                'opt_vq_decoder': self.opt_vq_decoder.state_dict(),

                'ep': ep,
                'total_it': total_it,
            }
        torch.save(state, file_name)

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        self.vq_encoder.load_state_dict(checkpoint['vq_encoder'])
        self.quantizer.load_state_dict(checkpoint['quantizer'])
        self.vq_decoder.load_state_dict(checkpoint['vq_decoder'])

        self.opt_vq_encoder.load_state_dict(checkpoint['opt_vq_encoder'])
        self.opt_quantizer.load_state_dict(checkpoint['opt_quantizer'])
        self.opt_vq_decoder.load_state_dict(checkpoint['opt_vq_decoder'])

        # if self.opt.use_gan:
        #     self.discriminator.load_state_dict(checkpoint['discriminator'])
        #     self.opt_discriminator.load_state_dict(checkpoint['opt_discriminator'])
        return checkpoint['ep'], checkpoint['total_it']

    def train(self, train_dataloader, val_dataloader, plot_eval):
        self.vq_encoder.to(self.device)
        self.quantizer.to(self.device)
        self.vq_decoder.to(self.device)
        if self.opt.use_percep:
            self.mov_encoder.to(self.device)

        self.opt_vq_encoder = optim.Adam(self.vq_encoder.parameters(), lr=self.opt.lr)
        self.opt_quantizer = optim.Adam(self.quantizer.parameters(), lr=self.opt.lr)
        self.opt_vq_decoder = optim.Adam(self.vq_decoder.parameters(), lr=self.opt.lr)

        if self.opt.use_gan:
            self.discriminator.to(self.device)
            self.opt_discriminator = optim.Adam(self.discriminator.parameters(), lr=self.opt.lr)

        epoch = 0
        it = 0
        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')
            epoch, it = self.resume(model_dir)

        start_time = time.time()
        total_iters = self.opt.max_epoch * len(train_dataloader)
        print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_dataloader), len(val_dataloader)))
        val_loss = 0
        min_val_loss = np.inf
        min_val_epoch = epoch
        logs = OrderedDict()
        while epoch < self.opt.max_epoch:
            self.opt.start_use_gan = self.opt.use_gan & (epoch > self.opt.start_dis_epoch)
            for i, batch_data in enumerate(train_dataloader):
                self.vq_encoder.train()
                self.quantizer.train()
                self.vq_decoder.train()
                if self.opt.use_percep:
                    self.mov_encoder.train()
                if self.opt.start_use_gan:
                    # print('Introducing Adversarial Loss!~')
                    self.discriminator.train()

                self.forward(batch_data)

                log_dict = self.update()
                # time3 = time.time()
                # print('Update Time: %.5f s' % (time3 - time2))
                # time0 = time3
                for k, v in log_dict.items():
                    if k not in logs:
                        logs[k] = v
                    else:
                        logs[k] += v

                it += 1
                if it % self.opt.log_every == 0:
                    mean_loss = OrderedDict({'val_loss': val_loss})
                    self.logger.scalar_summary('val_loss', val_loss, it)

                    for tag, value in logs.items():
                        self.logger.scalar_summary(tag, value / self.opt.log_every, it)
                        mean_loss[tag] = value / self.opt.log_every
                    logs = OrderedDict()
                    print_current_loss(start_time, it, total_iters, mean_loss, epoch, i)

                if it % self.opt.save_latest == 0:
                    self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            epoch += 1
            if epoch % self.opt.save_every_e == 0:
                self.save(pjoin(self.opt.model_dir, 'E%04d.tar' % (epoch)), epoch, total_it=it)

            print('Validation time:')

            val_loss_rec = 0
            val_loss_emb = 0
            with torch.no_grad():
                for i, batch_data in enumerate(val_dataloader):
                    self.forward(batch_data)
                    val_loss_rec += self.l1_criterion(self.recon_motions, self.motions).item()
                    # val_loss_emb += self.embedding_loss.item()

            val_loss = val_loss_rec / (len(val_dataloader) + 1)
            # val_loss = val_loss / (len(val_dataloader) + 1)
            # val_loss_rec = val_loss_rec / (len(val_dataloader) + 1)
            # val_loss_emb = val_loss_emb / (len(val_dataloader) + 1)

            print('Validation Loss: %.5f' % (val_loss))

            if val_loss < min_val_loss:
                min_val_loss = val_loss
                min_val_epoch = epoch
                self.save(pjoin(self.opt.model_dir, 'finest.tar'), epoch, it)
                print('Best Validation Model So Far!~')

            if epoch % self.opt.eval_every_e == 0:
                data = torch.cat([self.recon_motions[:4], self.motions[:4]], dim=0).detach().cpu().numpy()
                save_dir = pjoin(self.opt.eval_dir, 'E%04d' % (epoch))
                os.makedirs(save_dir, exist_ok=True)
                plot_eval(data, save_dir)

            if epoch - min_val_epoch >= 5:
                print('Early Stopping!~')
                # break


class VQTokenizerTrainerV3(Trainer):
    def __init__(self, args, vq_encoder, quantizer, vq_decoder, discriminator=None):
        self.opt = args
        self.vq_encoder = vq_encoder
        self.vq_decoder = vq_decoder
        self.quantizer = quantizer
        # self.mov_encoder = mov_encoder
        self.discriminator = discriminator
        self.device = args.device

        if args.is_train:
            self.logger = Logger(args.log_dir)
            self.l1_criterion = torch.nn.L1Loss()
            self.gan_criterion = torch.nn.BCEWithLogitsLoss()
            self.disc_loss = self.hinge_d_loss

    def hinge_d_loss(self, logits_real, logits_fake):
        loss_real = torch.mean(F.relu(1. - logits_real))
        loss_fake = torch.mean(F.relu(1. + logits_fake))
        d_loss = 0.5 * (loss_real + loss_fake)
        return d_loss

    # def ones_like(self, tensor, val=1.):
    #     return torch.FloatTensor(tensor.size()).fill_(val).to(self.opt.gpu_id)
    #
    # # @staticmethod
    # def zeros_like(self, tensor, val=0.):
    #     return torch.FloatTensor(tensor.size()).fill_(val).to(self.opt.gpu_id)

    def forward(self, batch_data):
        motions = batch_data
        self.motions = motions.detach().to(self.device).float()
        # print(self.motions.shape)
        self.pre_latents = self.vq_encoder(self.motions[..., :-4])
        # print(self.pre_latents.shape)
        self.embedding_loss, self.vq_latents, _, self.perplexity = self.quantizer(self.pre_latents)
        # print(self.vq_latents.shape)
        self.recon_motions = self.vq_decoder(self.vq_latents)

    def calculate_adaptive_weight(self, rec_loss, gan_loss, last_layer):
        rec_grads = torch.autograd.grad(rec_loss, last_layer, retain_graph=True)[0]
        gan_grads = torch.autograd.grad(gan_loss, last_layer, retain_graph=True)[0]

        d_weight = torch.norm(rec_grads) / (torch.norm(gan_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.opt.lambda_adv
        return d_weight


    def backward_G(self):
        self.loss_rec_mot = self.l1_criterion(self.recon_motions, self.motions)
        self.loss_G = self.loss_rec_mot + self.embedding_loss

        if self.opt.start_use_gan:
            _, logits_fake = self.discriminator(self.recon_motions)
            self.loss_G_adv = -torch.mean(logits_fake)
            # last_layer = self.vq_decoder.main[9].weight
            #
            # try:
            #     self.d_weight = self.calculate_adaptive_weight(self.loss_rec_mot, self.loss_G_adv, last_layer=last_layer)
            # except RuntimeError:
            #     assert not self.opt.is_train
            #     self.d_weight = torch.tensor(0.0)
            # self.loss_G += self.d_weight * self.loss_G_adv
            self.loss_G += self.opt.lambda_adv * self.loss_G_adv


    def backward_D(self):
        self.real_feats, real_labels = self.discriminator(self.motions.detach())
        fake_feats, fake_labels = self.discriminator(self.recon_motions.detach())

        self.loss_D = self.disc_loss(real_labels, fake_labels) * self.opt.lambda_adv
        # self.loss_D = (self.loss_D_T + self.loss_D_F) * self.opt.lambda_adv


    def update(self):
        loss_logs = OrderedDict({})

        if self.opt.start_use_gan:
            self.zero_grad([self.opt_discriminator])
            self.backward_D()
            self.loss_D.backward(retain_graph=True)
            self.step([self.opt_discriminator])

        self.zero_grad([self.opt_vq_encoder, self.opt_quantizer, self.opt_vq_decoder])
        self.backward_G()
        self.loss_G.backward()
        self.step([self.opt_vq_encoder, self.opt_quantizer, self.opt_vq_decoder])

        loss_logs['loss_G'] = self.loss_G.item()
        loss_logs['loss_G_rec_mot'] = self.loss_rec_mot.item()
        loss_logs['loss_G_emb'] = self.embedding_loss.item()
        loss_logs['perplexity'] = self.perplexity.item()

        if self.opt.start_use_gan:
            # loss_logs['d_weight'] = self.d_weight.item()
            loss_logs['loss_G_adv'] = self.loss_G_adv.item()
            loss_logs['loss_D'] = self.loss_D.item()

        return loss_logs

    def save(self, file_name, ep, total_it):
        state = {
            'vq_encoder': self.vq_encoder.state_dict(),
            'quantizer': self.quantizer.state_dict(),
            'vq_decoder': self.vq_decoder.state_dict(),
            'discriminator': self.discriminator.state_dict(),

            'opt_vq_encoder': self.opt_vq_encoder.state_dict(),
            'opt_quantizer': self.opt_quantizer.state_dict(),
            'opt_vq_decoder': self.opt_vq_decoder.state_dict(),
            'opt_discriminator': self.opt_discriminator.state_dict(),

            'ep': ep,
            'total_it': total_it,
        }
        torch.save(state, file_name)

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        self.vq_encoder.load_state_dict(checkpoint['vq_encoder'])
        self.quantizer.load_state_dict(checkpoint['quantizer'])
        self.vq_decoder.load_state_dict(checkpoint['vq_decoder'])

        self.opt_vq_encoder.load_state_dict(checkpoint['opt_vq_encoder'])
        self.opt_quantizer.load_state_dict(checkpoint['opt_quantizer'])
        self.opt_vq_decoder.load_state_dict(checkpoint['opt_vq_decoder'])

        # if self.opt.use_gan:
        self.discriminator.load_state_dict(checkpoint['discriminator'])
        self.opt_discriminator.load_state_dict(checkpoint['opt_discriminator'])
        return checkpoint['ep'], checkpoint['total_it']

    def train(self, train_dataloader, val_dataloader, plot_eval):
        self.vq_encoder.to(self.device)
        self.quantizer.to(self.device)
        self.vq_decoder.to(self.device)
        self.discriminator.to(self.device)

        self.opt_vq_encoder = optim.Adam(self.vq_encoder.parameters(), lr=self.opt.lr)
        self.opt_quantizer = optim.Adam(self.quantizer.parameters(), lr=self.opt.lr)
        self.opt_vq_decoder = optim.Adam(self.vq_decoder.parameters(), lr=self.opt.lr)
        self.opt_discriminator = optim.Adam(self.discriminator.parameters(), lr=self.opt.lr)

        epoch = 0
        it = 0
        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')
            epoch, it = self.resume(model_dir)
            print("Load model epoch:%d iterations:%d"%(epoch, it))

        start_time = time.time()
        total_iters = self.opt.max_epoch * len(train_dataloader)
        print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_dataloader), len(val_dataloader)))
        val_loss = 0
        min_val_loss = np.inf
        min_val_epoch = epoch
        logs = OrderedDict()
        while epoch < self.opt.max_epoch:
            self.opt.start_use_gan = (epoch >= self.opt.start_dis_epoch)
            for i, batch_data in enumerate(train_dataloader):
                self.vq_encoder.train()
                self.quantizer.train()
                self.vq_decoder.train()
                # if self.opt.use_percep:
                #     self.mov_encoder.train()
                if self.opt.start_use_gan:
                    # print('Introducing Adversarial Loss!~')
                    self.discriminator.train()

                self.forward(batch_data)

                log_dict = self.update()
                # time3 = time.time()
                # print('Update Time: %.5f s' % (time3 - time2))
                # time0 = time3
                for k, v in log_dict.items():
                    if k not in logs:
                        logs[k] = v
                    else:
                        logs[k] += v

                it += 1
                if it % self.opt.log_every == 0:
                    mean_loss = OrderedDict({'val_loss': val_loss})
                    self.logger.scalar_summary('val_loss', val_loss, it)

                    for tag, value in logs.items():
                        self.logger.scalar_summary(tag, value / self.opt.log_every, it)
                        mean_loss[tag] = value / self.opt.log_every
                    logs = OrderedDict()
                    print_current_loss(start_time, it, total_iters, mean_loss, epoch, i)

                if it % self.opt.save_latest == 0:
                    self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            epoch += 1
            if epoch % self.opt.save_every_e == 0:
                self.save(pjoin(self.opt.model_dir, 'E%04d.tar' % (epoch)), epoch, total_it=it)

            print('Validation time:')

            val_loss_rec = 0
            val_loss_emb = 0
            with torch.no_grad():
                for i, batch_data in enumerate(val_dataloader):
                    self.forward(batch_data)
                    val_loss_rec += self.l1_criterion(self.recon_motions, self.motions).item()
                    # val_loss_emb += self.embedding_loss.item()

            val_loss = val_loss_rec / (len(val_dataloader) + 1)
            # val_loss = val_loss / (len(val_dataloader) + 1)
            # val_loss_rec = val_loss_rec / (len(val_dataloader) + 1)
            # val_loss_emb = val_loss_emb / (len(val_dataloader) + 1)

            print('Validation Loss: %.5f' % (val_loss))

            if val_loss < min_val_loss:
                min_val_loss = val_loss
                min_val_epoch = epoch
                self.save(pjoin(self.opt.model_dir, 'finest.tar'), epoch, it)
                print('Best Validation Model So Far!~')

            if epoch % self.opt.eval_every_e == 0:
                data = torch.cat([self.recon_motions[:4], self.motions[:4]], dim=0).detach().cpu().numpy()
                save_dir = pjoin(self.opt.eval_dir, 'E%04d' % (epoch))
                os.makedirs(save_dir, exist_ok=True)
                plot_eval(data, save_dir)

            if epoch - min_val_epoch >= 5:
                print('Early Stopping!~')
                # break


class TransformerT2MTrainer(Trainer):
    def __init__(self, args, t2m_transformer):
        self.opt = args
        self.t2m_transformer = t2m_transformer
        # self.quantizer = quantizer
        # self.vq_decoder = vq_decoder
        self.device = args.device

        # self.trg_pad_index = args.trg_pad_index
        # self.trg_start_index = args.trg_start_index
        # self.trg_end_index = args.trg_end_index
        # self.trg_num_vocab = args.trg_num_vocab

        if args.is_train:
            self.logger = Logger(args.log_dir)

    def forward(self, batch_data):
        word_emb, word_tokens, caption, cap_lens, m_tokens, _ = batch_data
        word_emb = word_emb.detach().to(self.device).float()
        # pos_ohot = pos_ohot.detach().to(self.device).float()
        m_tokens = m_tokens.detach().to(self.device).long()
        word_tokens = word_tokens.detach().to(self.device).long()

        self.cap_lens = cap_lens
        self.caption = caption

        trg_input, self.gold = m_tokens[:, :-1], m_tokens[:, 1:]

        if self.opt.t2m_v2:
            self.trg_pred = self.t2m_transformer(word_tokens, trg_input)
        else:
            self.trg_pred = self.t2m_transformer(word_emb, trg_input, cap_lens)

        # one_hot_indices = F.one_hot(encoding_indices, num_classes=self.args.trg_num_vocab)

    def backward(self):
        # print(self.trg_pred.shape, self.gold.shape)
        trg_pred = self.trg_pred.view(-1, self.trg_pred.shape[-1]).clone()
        # print(trg_pred[0])
        gold = self.gold.contiguous().view(-1).clone()
        self.loss, self.pred_seq, self.n_correct, self.n_word = cal_performance(trg_pred, gold, self.opt.mot_pad_idx,
                                                            smoothing=self.opt.label_smoothing)
        # print(gold, self.pred_seq)
        # self.loss = loss / n_word
        loss_logs = OrderedDict({})
        loss_logs['loss'] = self.loss.item() / self.n_word
        loss_logs['accuracy'] = self.n_correct / self.n_word

        return loss_logs

    def update(self):
        self.zero_grad([self.opt_t2m_transformer])
        # time2_0 = time.time()
        # print("\t\t Zero Grad:%5f" % (time2_0 - time1))
        loss_logs = self.backward()
        self.loss.backward()

        # time2_3 = time.time()
        # print("\t\t Clip Norm :%5f" % (time2_3 - time2_2))
        self.step([self.opt_t2m_transformer])

        return loss_logs

    def save(self, file_name, ep, total_it):

        state = {
            't2m_transformer': self.t2m_transformer.state_dict(),

            'opt_t2m_transformer': self.opt_t2m_transformer.state_dict(),

            'ep': ep,
            'total_it': total_it,
        }
        torch.save(state, file_name)

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        self.t2m_transformer.load_state_dict(checkpoint['t2m_transformer'])

        self.opt_t2m_transformer.load_state_dict(checkpoint['opt_t2m_transformer'])
        # if self.opt.use_gan:
        #     self.discriminator.load_state_dict(checkpoint['discriminator'])
        #     self.opt_discriminator.load_state_dict(checkpoint['opt_discriminator'])
        return checkpoint['ep'], checkpoint['total_it']


    def train(self, train_dataloader, val_dataloader, plot_eval):
        self.t2m_transformer.to(self.device)
        # self.vq_decoder.to(self.device)
        # self.quantizer.to(self.device)

        self.opt_t2m_transformer = optim.Adam(self.t2m_transformer.parameters(), lr=self.opt.lr)


        epoch = 0
        it = 0
        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')
            epoch, it = self.resume(model_dir)

        start_time = time.time()
        total_iters = self.opt.max_epoch * len(train_dataloader)
        print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_dataloader), len(val_dataloader)))
        val_loss = 0
        val_accuracy = 0
        min_val_loss = np.inf
        min_val_epoch = epoch
        logs = OrderedDict()
        while epoch < self.opt.max_epoch:
            for i, batch_data in enumerate(train_dataloader):
                self.t2m_transformer.train()

                self.forward(batch_data)

                log_dict = self.update()
                # continue
                # time3 = time.time()
                # print('Update Time: %.5f s' % (time3 - time2))
                # time0 = time3
                for k, v in log_dict.items():
                    if k not in logs:
                        logs[k] = v
                    else:
                        logs[k] += v

                it += 1
                if it % self.opt.log_every == 0:
                    mean_loss = OrderedDict({'val_loss': val_loss, 'val_accuracy':val_accuracy})
                    self.logger.scalar_summary('val_loss', val_loss, it)
                    self.logger.scalar_summary('val_accuracy', val_accuracy, it)

                    for tag, value in logs.items():
                        self.logger.scalar_summary(tag, value / self.opt.log_every, it)
                        mean_loss[tag] = value / self.opt.log_every
                    logs = OrderedDict()
                    print_current_loss(start_time, it, total_iters, mean_loss, epoch, i)

                if it % self.opt.save_latest == 0:
                    self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            epoch += 1
            if epoch % self.opt.save_every_e == 0:
                self.save(pjoin(self.opt.model_dir, 'E%04d.tar' % (epoch)), epoch, total_it=it)

            print('Validation time:')

            val_loss = 0
            val_accuracy = 0
            with torch.no_grad():
                for i, batch_data in enumerate(val_dataloader):
                    self.forward(batch_data)
                    self.backward()
                    val_loss += self.loss.item() / self.n_word
                    val_accuracy += self.n_correct / self.n_word
                    # val_loss_rec += self.l1_criterion(self.recon_motions, self.motions).item()
                    # val_loss_emb += self.embedding_loss.item()

            val_loss = val_loss / len(val_dataloader)
            val_accuracy = val_accuracy / len(val_dataloader)
            # val_loss = val_loss / (len(val_dataloader) + 1)
            # val_loss_rec = val_loss_rec / (len(val_dataloader) + 1)
            # val_loss_emb = val_loss_emb / (len(val_dataloader) + 1)
            print(self.gold[0])
            print(self.pred_seq.view(self.gold.shape)[0])

            print('Validation Loss: %.5f Validation Accuracy: %.4f' % (val_loss, val_accuracy))

            if val_loss < min_val_loss:
                min_val_loss = val_loss
                min_val_epoch = epoch
                self.save(pjoin(self.opt.model_dir, 'finest.tar'), epoch, it)
                print('Best Validation Model So Far!~')

            # if epoch % self.opt.eval_every_e == 0:
                # self.quantizer.eval()
                # self.vq_decoder.eval()
                # with torch.no_grad():
                #     pred_seq = self.pred_seq.view(self.gold.shape)[0:1]
                #     # print(pred_seq.shape)
                #     non_pad_mask = self.gold[0:1].ne(self.opt.trg_pad_idx)
                #     pred_seq = pred_seq.masked_select(non_pad_mask).unsqueeze(0)
                #     # print(non_pad_mask.shape)
                #     # print(pred_seq.shape)
                #     # print(pred_seq)
                #     # print(self.gold[0:1])
                #     vq_latent = self.quantizer.get_codebook_entry(pred_seq)
                #     # print(vq_latent.shape)
                #
                #     rec_motion = self.vq_decoder(vq_latent)
                #
                # save_dir = pjoin(self.opt.eval_dir, 'E%04d' % (epoch))
                # os.makedirs(save_dir, exist_ok=True)
                # plot_eval(rec_motion.detach().cpu().numpy(), self.caption[0:1], save_dir)
                # save_dir = pjoin(self.opt.eval_dir, 'E%04d' % epoch)
                # os.makedirs()

            if epoch - min_val_epoch >= 5:
                print('Early Stopping!~')
                # break


class TransformerM2TTrainer(Trainer):
    def __init__(self, args, m2t_transformer):
        self.opt = args
        self.m2t_transformer = m2t_transformer
        # self.quantizer = quantizer
        # self.vq_decoder = vq_decoder
        self.device = args.device

        # self.trg_pad_index = args.trg_pad_index
        # self.trg_start_index = args.trg_start_index
        # self.trg_end_index = args.trg_end_index
        # self.trg_num_vocab = args.trg_num_vocab

        if args.is_train:
            self.logger = Logger(args.log_dir)

    def forward(self, batch_data):
        word_emb, word_tokens, caption, cap_lens, m_tokens, _ = batch_data
        word_emb = word_emb.detach().to(self.device).float()
        # pos_ohot = pos_ohot.detach().to(self.device).float()
        m_tokens = m_tokens.detach().to(self.device).long()
        word_tokens = word_tokens.detach().to(self.device).long()

        cap_lens = cap_lens - 1
        self.cap_lens = cap_lens
        self.caption = caption

        self.gold = word_tokens[:, 1:]

        """Input pretrained word vector"""
        if self.opt.m2t_v3:
            trg_input = word_emb[:, :-1]
            self.trg_pred = self.m2t_transformer(m_tokens, trg_input, cap_lens)
        else:
            trg_input = word_tokens[:, :-1]
            self.trg_pred = self.m2t_transformer(m_tokens, trg_input)

        # one_hot_indices = F.one_hot(encoding_indices, num_classes=self.args.trg_num_vocab)

    def backward(self):
        # print(self.trg_pred.shape, self.gold.shape)
        trg_pred = self.trg_pred.view(-1, self.trg_pred.shape[-1]).clone()
        # print(trg_pred[0])
        gold = self.gold.contiguous().view(-1).clone()
        self.loss, self.pred_seq, self.n_correct, self.n_word = cal_performance(trg_pred, gold, self.opt.txt_pad_idx,
                                                            smoothing=self.opt.label_smoothing)
        # print(gold, self.pred_seq)
        # self.loss = loss / n_word
        loss_logs = OrderedDict({})
        loss_logs['loss'] = self.loss.item() / self.n_word
        loss_logs['accuracy'] = self.n_correct / self.n_word

        return loss_logs

    def update(self):
        self.zero_grad([self.opt_m2t_transformer])
        # time2_0 = time.time()
        # print("\t\t Zero Grad:%5f" % (time2_0 - time1))
        loss_logs = self.backward()
        self.loss.backward()

        # time2_3 = time.time()
        # print("\t\t Clip Norm :%5f" % (time2_3 - time2_2))
        self.step([self.opt_m2t_transformer])

        return loss_logs

    def save(self, file_name, ep, total_it):

        state = {
            'm2t_transformer': self.m2t_transformer.state_dict(),

            'opt_m2t_transformer': self.opt_m2t_transformer.state_dict(),

            'ep': ep,
            'total_it': total_it,
        }
        torch.save(state, file_name)

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        self.m2t_transformer.load_state_dict(checkpoint['m2t_transformer'])

        self.opt_m2t_transformer.load_state_dict(checkpoint['opt_m2t_transformer'])
        # if self.opt.use_gan:
        #     self.discriminator.load_state_dict(checkpoint['discriminator'])
        #     self.opt_discriminator.load_state_dict(checkpoint['opt_discriminator'])
        return checkpoint['ep'], checkpoint['total_it']


    def train(self, train_dataloader, val_dataloader, w_vectorizer):
        self.m2t_transformer.to(self.device)
        # self.vq_decoder.to(self.device)
        # self.quantizer.to(self.device)

        self.opt_m2t_transformer = optim.Adam(self.m2t_transformer.parameters(), lr=self.opt.lr)


        epoch = 0
        it = 0
        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')
            epoch, it = self.resume(model_dir)

        start_time = time.time()
        total_iters = self.opt.max_epoch * len(train_dataloader)
        print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_dataloader), len(val_dataloader)))
        val_loss = 0
        val_accuracy = 0
        min_val_loss = np.inf
        min_val_epoch = epoch
        logs = OrderedDict()
        while epoch < self.opt.max_epoch:
            for i, batch_data in enumerate(train_dataloader):
                self.m2t_transformer.train()

                self.forward(batch_data)

                log_dict = self.update()
                # continue
                # time3 = time.time()
                # print('Update Time: %.5f s' % (time3 - time2))
                # time0 = time3
                for k, v in log_dict.items():
                    if k not in logs:
                        logs[k] = v
                    else:
                        logs[k] += v

                it += 1
                if it % self.opt.log_every == 0:
                    mean_loss = OrderedDict({'val_loss': val_loss, 'val_accuracy':val_accuracy})
                    self.logger.scalar_summary('val_loss', val_loss, it)
                    self.logger.scalar_summary('val_accuracy', val_accuracy, it)

                    for tag, value in logs.items():
                        self.logger.scalar_summary(tag, value / self.opt.log_every, it)
                        mean_loss[tag] = value / self.opt.log_every
                    logs = OrderedDict()
                    print_current_loss(start_time, it, total_iters, mean_loss, epoch, i)

                if it % self.opt.save_latest == 0:
                    self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            epoch += 1
            if epoch % self.opt.save_every_e == 0:
                self.save(pjoin(self.opt.model_dir, 'E%04d.tar' % (epoch)), epoch, total_it=it)

            print('Validation time:')

            val_loss = 0
            val_accuracy = 0
            with torch.no_grad():
                for i, batch_data in enumerate(val_dataloader):
                    self.forward(batch_data)
                    self.backward()
                    val_loss += self.loss.item() / self.n_word
                    val_accuracy += self.n_correct / self.n_word
                    # val_loss_rec += self.l1_criterion(self.recon_motions, self.motions).item()
                    # val_loss_emb += self.embedding_loss.item()

            val_loss = val_loss / len(val_dataloader)
            val_accuracy = val_accuracy / len(val_dataloader)
            # val_loss = val_loss / (len(val_dataloader) + 1)
            # val_loss_rec = val_loss_rec / (len(val_dataloader) + 1)
            # val_loss_emb = val_loss_emb / (len(val_dataloader) + 1)

            print('Validation Loss: %.5f Validation Accuracy: %.4f' % (val_loss, val_accuracy))

            if val_loss < min_val_loss:
                min_val_loss = val_loss
                min_val_epoch = epoch
                self.save(pjoin(self.opt.model_dir, 'finest.tar'), epoch, it)
                print('Best Validation Model So Far!~')

            print(self.gold[0])
            print(self.pred_seq.view(self.gold.shape)[0])
            gt_seq = ' '.join(w_vectorizer.itos(i) for i in self.gold[0, :self.cap_lens[0]].cpu().numpy())
            pred_seq = ' '.join(
                w_vectorizer.itos(i) for i in self.pred_seq.view(self.gold.shape)[0, :self.cap_lens[0]].cpu().numpy())
            print(gt_seq)
            print(pred_seq)
            # if epoch % self.opt.eval_every_e == 0:
                # self.quantizer.eval()
                # self.vq_decoder.eval()
                # with torch.no_grad():
                #     pred_seq = self.pred_seq.view(self.gold.shape)[0:1]
                #     # print(pred_seq.shape)
                #     non_pad_mask = self.gold[0:1].ne(self.opt.trg_pad_idx)
                #     pred_seq = pred_seq.masked_select(non_pad_mask).unsqueeze(0)
                #     # print(non_pad_mask.shape)
                #     # print(pred_seq.shape)
                #     # print(pred_seq)
                #     # print(self.gold[0:1])
                #     vq_latent = self.quantizer.get_codebook_entry(pred_seq)
                #     # print(vq_latent.shape)
                #
                #     rec_motion = self.vq_decoder(vq_latent)
                #
                # save_dir = pjoin(self.opt.eval_dir, 'E%04d' % (epoch))
                # os.makedirs(save_dir, exist_ok=True)
                # plot_eval(rec_motion.detach().cpu().numpy(), self.caption[0:1], save_dir)
                # save_dir = pjoin(self.opt.eval_dir, 'E%04d' % epoch)
                # os.makedirs()

            if epoch - min_val_epoch >= 5:
                print('Early Stopping!~')
                # break


class TransformerT2MJointTrainer(Trainer):
    def __init__(self, args, t2m_transformer, m2t_transformer):
        self.opt = args
        self.t2m_transformer = t2m_transformer
        # self.quantizer = quantizer
        # self.vq_decoder = vq_decoder
        self.m2t_transformer = m2t_transformer
        self.device = args.device

        # self.trg_pad_index = args.trg_pad_index
        # self.trg_start_index = args.trg_start_index
        # self.trg_end_index = args.trg_end_index
        # self.trg_num_vocab = args.trg_num_vocab

        if args.is_train:
            self.logger = Logger(args.log_dir)

    def forward(self, batch_data):
        word_emb, word_tokens, caption, cap_lens, m_tokens, _ = batch_data
        word_emb = word_emb.detach().to(self.device).float()
        # pos_ohot = pos_ohot.detach().to(self.device).float()
        m_tokens = m_tokens.detach().to(self.device).long()
        word_tokens = word_tokens.detach().to(self.device).long()

        self.cap_lens = cap_lens
        self.caption = caption

        mot_input, self.mot_gold = m_tokens[:, :-1], m_tokens[:, 1:]

        if self.opt.t2m_v2:
            self.mot_pred_prob = self.t2m_transformer(word_tokens, mot_input)
        else:
            self.mot_pred_prob = self.t2m_transformer(word_emb, mot_input, cap_lens)

        mot_pred_ohot = F.gumbel_softmax(self.mot_pred_prob, tau=1.0, hard=True)
        mot_start_ohot = F.one_hot(m_tokens[:, 0:1], num_classes=self.opt.n_mot_vocab)
        mot_pred_seq_ohot = torch.cat([mot_start_ohot, mot_pred_ohot], dim=1)
        mot_mask = get_pad_mask_idx(m_tokens, self.opt.mot_pad_idx)

        txt_input, self.txt_gold = word_tokens[:, :-1], word_tokens[:, 1:]
        self.txt_pred_prob = self.m2t_transformer(mot_pred_seq_ohot, txt_input, input_onehot=True, src_mask=mot_mask)
        # forward(self, src_seq, trg_seq, input_onehot=False, src_mask=None):

    def backward(self):
        # print(self.trg_pred.shape, self.gold.shape)
        mot_pred = self.mot_pred_prob.view(-1, self.mot_pred_prob.shape[-1]).clone()
        # print(trg_pred[0])
        mot_gold = self.mot_gold.contiguous().view(-1).clone()
        self.t2m_loss, self.mot_pred_seq, self.mot_n_correct, self.mot_n_word = cal_performance(mot_pred, mot_gold,
                                                                                                self.opt.mot_pad_idx,
                                                                                                smoothing=self.opt.label_smoothing)

        txt_pred = self.txt_pred_prob.view(-1, self.txt_pred_prob.shape[-1]).clone()
        txt_gold = self.txt_gold.contiguous().view(-1).clone()
        self.m2t_loss, self.txt_pred_seq, self.txt_n_correct, self.txt_n_word = cal_performance(txt_pred, txt_gold,
                                                                                                self.opt.txt_pad_idx,
                                                                                                smoothing=self.opt.label_smoothing)
        self.loss = self.t2m_loss + self.opt.lambda_m2t * self.m2t_loss
        # print(gold, self.pred_seq)
        # self.loss = loss / n_word
        loss_logs = OrderedDict({})
        # loss_logs['loss'] = self.loss.item() /
        loss_logs['t2m_loss'] = self.t2m_loss.item() / self.mot_n_word
        loss_logs['t2m_accuracy'] = self.mot_n_correct / self.mot_n_word

        loss_logs['m2t_loss'] = self.m2t_loss.item() / self.txt_n_word
        loss_logs['m2t_accuracy'] = self.txt_n_correct / self.txt_n_word

        return loss_logs

    def update(self):
        self.zero_grad([self.opt_t2m_transformer, self.opt_m2t_transformer])
        # time2_0 = time.time()
        # print("\t\t Zero Grad:%5f" % (time2_0 - time1))
        loss_logs = self.backward()
        self.loss.backward()

        # time2_3 = time.time()
        # print("\t\t Clip Norm :%5f" % (time2_3 - time2_2))
        self.step([self.opt_t2m_transformer])

        return loss_logs

    def save(self, file_name, ep, total_it):

        state = {
            't2m_transformer': self.t2m_transformer.state_dict(),

            'opt_t2m_transformer': self.opt_t2m_transformer.state_dict(),

            'ep': ep,
            'total_it': total_it,
        }
        torch.save(state, file_name)

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        self.t2m_transformer.load_state_dict(checkpoint['t2m_transformer'])

        self.opt_t2m_transformer.load_state_dict(checkpoint['opt_t2m_transformer'])
        # if self.opt.use_gan:
        #     self.discriminator.load_state_dict(checkpoint['discriminator'])
        #     self.opt_discriminator.load_state_dict(checkpoint['opt_discriminator'])
        return checkpoint['ep'], checkpoint['total_it']

    def train(self, train_dataloader, val_dataloader, w_vectorizer):
        self.t2m_transformer.to(self.device)
        self.m2t_transformer.to(self.device)
        # self.vq_decoder.to(self.device)
        # self.quantizer.to(self.device)

        self.opt_t2m_transformer = optim.Adam(self.t2m_transformer.parameters(), lr=self.opt.lr)


        epoch = 0
        it = 0
        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')
            epoch, it = self.resume(model_dir)

        start_time = time.time()
        total_iters = self.opt.max_epoch * len(train_dataloader)
        print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_dataloader), len(val_dataloader)))
        val_loss = 0
        val_accuracy = 0
        min_val_loss = np.inf
        min_val_epoch = epoch
        logs = OrderedDict()
        while epoch < self.opt.max_epoch:
            for i, batch_data in enumerate(train_dataloader):
                self.t2m_transformer.train()
                self.m2t_transformer.train()

                self.forward(batch_data)

                log_dict = self.update()
                # continue
                # time3 = time.time()
                # print('Update Time: %.5f s' % (time3 - time2))
                # time0 = time3
                for k, v in log_dict.items():
                    if k not in logs:
                        logs[k] = v
                    else:
                        logs[k] += v

                it += 1
                if it % self.opt.log_every == 0:
                    mean_loss = OrderedDict({'val_loss': val_loss, 'val_accuracy':val_accuracy})
                    self.logger.scalar_summary('val_loss', val_loss, it)
                    self.logger.scalar_summary('val_accuracy', val_accuracy, it)

                    for tag, value in logs.items():
                        self.logger.scalar_summary(tag, value / self.opt.log_every, it)
                        mean_loss[tag] = value / self.opt.log_every
                    logs = OrderedDict()
                    print_current_loss(start_time, it, total_iters, mean_loss, epoch, i)

                if it % self.opt.save_latest == 0:
                    self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            epoch += 1
            if epoch % self.opt.save_every_e == 0:
                self.save(pjoin(self.opt.model_dir, 'E%04d.tar' % (epoch)), epoch, total_it=it)

            print('Validation time:')

            val_loss = 0
            val_accuracy = 0
            val_m2t_loss = 0
            val_m2t_accuracy = 0
            with torch.no_grad():
                for i, batch_data in enumerate(val_dataloader):
                    self.forward(batch_data)
                    self.backward()
                    val_loss += self.t2m_loss.item() / self.mot_n_word
                    val_accuracy += self.mot_n_correct / self.mot_n_word
                    val_m2t_loss += self.m2t_loss.item() / self.txt_n_word
                    val_m2t_accuracy += self.txt_n_correct / self.txt_n_word
                    # val_loss_rec += self.l1_criterion(self.recon_motions, self.motions).item()
                    # val_loss_emb += self.embedding_loss.item()

            val_loss = val_loss / len(val_dataloader)
            val_accuracy = val_accuracy / len(val_dataloader)
            val_m2t_loss = val_m2t_loss / len(val_dataloader)
            val_m2t_accuracy = val_m2t_accuracy / len(val_dataloader)
            # val_loss = val_loss / (len(val_dataloader) + 1)
            # val_loss_rec = val_loss_rec / (len(val_dataloader) + 1)
            # val_loss_emb = val_loss_emb / (len(val_dataloader) + 1)
            print(self.mot_gold[0])
            print(self.mot_pred_seq.view(self.mot_gold.shape)[0])

            gt_m2t_seq = ' '.join(w_vectorizer.itos(i) for i in self.txt_gold[0, :self.cap_lens[0]].cpu().numpy())
            pred_m2t_seq = ' '.join(
                w_vectorizer.itos(i) for i in self.txt_pred_seq.view(self.txt_gold.shape)[0, :self.cap_lens[0]].cpu().numpy())
            print(gt_m2t_seq)
            print(pred_m2t_seq)

            print('Validation Loss: %.5f Validation Accuracy: %.4f Validation(M2T) Loss: %.5f Validation(M2T) Accuracy: %.4f ' %
                  (val_loss, val_accuracy, val_m2t_loss, val_m2t_accuracy))

            if val_loss < min_val_loss:
                min_val_loss = val_loss
                min_val_epoch = epoch
                self.save(pjoin(self.opt.model_dir, 'finest.tar'), epoch, it)
                print('Best Validation Model So Far!~')

            # if epoch % self.opt.eval_every_e == 0:
                # self.quantizer.eval()
                # self.vq_decoder.eval()
                # with torch.no_grad():
                #     pred_seq = self.pred_seq.view(self.gold.shape)[0:1]
                #     # print(pred_seq.shape)
                #     non_pad_mask = self.gold[0:1].ne(self.opt.trg_pad_idx)
                #     pred_seq = pred_seq.masked_select(non_pad_mask).unsqueeze(0)
                #     # print(non_pad_mask.shape)
                #     # print(pred_seq.shape)
                #     # print(pred_seq)
                #     # print(self.gold[0:1])
                #     vq_latent = self.quantizer.get_codebook_entry(pred_seq)
                #     # print(vq_latent.shape)
                #
                #     rec_motion = self.vq_decoder(vq_latent)
                #
                # save_dir = pjoin(self.opt.eval_dir, 'E%04d' % (epoch))
                # os.makedirs(save_dir, exist_ok=True)
                # plot_eval(rec_motion.detach().cpu().numpy(), self.caption[0:1], save_dir)
                # save_dir = pjoin(self.opt.eval_dir, 'E%04d' % epoch)
                # os.makedirs()

            if epoch - min_val_epoch >= 5:
                print('Early Stopping!~')
                # break


class Seq2SeqT2MJointTrainer(Trainer):
    def __init__(self, args, t2m_model, m2t_model=None):
        self.opt = args
        self.t2m_model = t2m_model
        self.m2t_model = m2t_model
        # self.quantizer = quantizer
        # self.vq_decoder = vq_decoder
        self.device = args.device

        if args.is_train:
            self.logger = Logger(args.log_dir)

    def forward(self, batch_data, eval_mode=False):
        word_emb, word_tokens, caption, cap_lens, m_tokens, _ = batch_data
        word_emb = word_emb.detach().to(self.device).float()
        m_tokens = m_tokens.detach().to(self.device).long()

        self.mot_gold = m_tokens[:, 1:]

        trg_input = m_tokens[:, :-1]
        if eval_mode:
            self.mot_pred_prob = self.t2m_model(word_emb, trg_input, cap_lens, 0)
        else:
            self.mot_pred_prob = self.t2m_model(word_emb, trg_input, cap_lens, self.tf_ratio)

        # one_hot_indices = F.one_hot(encoding_indices, num_classes=self.args.trg_num_vocab)
        mot_pred = self.mot_pred_prob.view(-1, self.mot_pred_prob.shape[-1]).clone()
        # print(trg_pred[0])
        mot_gold = self.mot_gold.contiguous().view(-1).clone()
        self.t2m_loss, self.mot_pred_seq, self.mot_n_correct, self.mot_n_word = cal_performance(mot_pred, mot_gold,
                                                                                                self.opt.mot_pad_idx,
                                                                                                smoothing=self.opt.label_smoothing)
        # print(self.mot_pred_seq)

    def forward_cycle(self, batch_data):
        word_emb, word_tokens, caption, cap_lens, m_tokens, _ = batch_data
        self.cap_lens = cap_lens
        word_emb = word_emb.detach().to(self.device).float()
        word_tokens = word_tokens.detach().to(self.device).long()

        self.txt_gold = word_tokens[:, 1:]

        # self.mot_pred_prob = self.t2m_model(word_emb, trg_input, cap_lens, self.opt.tf_ratio)
        mot_pred_ohot, len_map = self.t2m_model.gumbel_sample(word_emb, cap_lens,
                                                              self.opt.mot_start_idx, self.opt.mot_end_idx,
                                                              top_k=self.opt.top_k)

        # print(len_map)
        len_map += 1

        txt_input, self.txt_gold = word_tokens[:, :-1], word_tokens[:, 1:]
        self.txt_pred_prob = self.m2t_model(mot_pred_ohot, txt_input, input_onehot=True,
                                            src_non_pad_lens=len_map.squeeze())

        txt_pred = self.txt_pred_prob.view(-1, self.txt_pred_prob.shape[-1]).clone()
        txt_gold = self.txt_gold.contiguous().view(-1).clone()
        self.m2t_loss, self.txt_pred_seq, self.txt_n_correct, self.txt_n_word = cal_performance(txt_pred, txt_gold,
                                                                                                self.opt.txt_pad_idx,
                                                                                                smoothing=self.opt.label_smoothing)

    def save(self, file_name, ep, total_it):

        state = {
            't2m_model': self.t2m_model.state_dict(),

            'opt_t2m_model': self.opt_t2m_model.state_dict(),

            'ep': ep,
            'total_it': total_it,
        }
        torch.save(state, file_name)

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        self.t2m_model.load_state_dict(checkpoint['t2m_model'])

        self.opt_t2m_model.load_state_dict(checkpoint['opt_t2m_model'])
        # if self.opt.use_gan:
        #     self.discriminator.load_state_dict(checkpoint['discriminator'])
        #     self.opt_discriminator.load_state_dict(checkpoint['opt_discriminator'])
        return checkpoint['ep'], checkpoint['total_it']

    def schedule_tf(self, epoch, start_tf=0.9, end_tf=0.1, end_epoch=35):
        tf_epoch = epoch if epoch < end_epoch else end_epoch
        return start_tf - (start_tf - end_tf) * tf_epoch / end_epoch

    def train(self, train_dataloader, val_dataloader, w_vectorizer):
        self.t2m_model.to(self.device)
        self.m2t_model.to(self.device)
        # self.vq_decoder.to(self.device)
        # self.quantizer.to(self.device)

        self.opt_t2m_model = optim.Adam(self.t2m_model.parameters(), lr=self.opt.lr)


        epoch = 0
        it = 0
        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')
            epoch, it = self.resume(model_dir)

        start_time = time.time()
        total_iters = self.opt.max_epoch * len(train_dataloader)
        print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_dataloader), len(val_dataloader)))
        val_loss = 0
        val_accuracy = 0
        min_val_loss = np.inf
        min_val_epoch = epoch
        logs = OrderedDict({
            "t2m_loss": 0,
            "t2m_acc": 0,
            "m2t_loss": 0,
            "m2t_acc": 0
        })
        start_m2t = False
        while epoch < self.opt.max_epoch:
            self.tf_ratio = self.schedule_tf(epoch)
            start_m2t = (epoch >= self.opt.start_m2t_ep)
            for i, batch_data in enumerate(train_dataloader):

                self.t2m_model.train()
                self.m2t_model.train()

                self.forward(batch_data)

                self.zero_grad([self.opt_t2m_model])
                self.t2m_loss.backward()
                self.clip_norm([self.t2m_model])
                self.step([self.opt_t2m_model])
                logs["t2m_loss"] += self.t2m_loss.item() / self.mot_n_word
                logs["t2m_acc"] += self.mot_n_correct / self.mot_n_word


                if start_m2t:
                    self.forward_cycle(batch_data)

                    self.zero_grad([self.opt_t2m_model])
                    self.m2t_loss.backward()
                    self.clip_norm([self.t2m_model])
                    self.step([self.opt_t2m_model])

                    logs["m2t_loss"] += self.m2t_loss.item() / self.txt_n_word
                    logs["m2t_acc"] += self.txt_n_correct / self.txt_n_word

                it += 1
                if it % self.opt.log_every == 0:
                    mean_loss = OrderedDict({'val_loss': val_loss, 'val_accuracy':val_accuracy})
                    self.logger.scalar_summary('val_loss', val_loss, it)
                    self.logger.scalar_summary('val_accuracy', val_accuracy, it)

                    for tag, value in logs.items():
                        self.logger.scalar_summary(tag, value / self.opt.log_every, it)
                        mean_loss[tag] = value / self.opt.log_every
                    logs = OrderedDict({
                        "t2m_loss": 0,
                        "t2m_acc": 0,
                        "m2t_loss": 0,
                        "m2t_acc": 0
                    })
                    print_current_loss(start_time, it, total_iters, mean_loss, epoch, i, tf_ratio=self.tf_ratio)

                if it % self.opt.save_latest == 0:
                    self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            epoch += 1
            if epoch % self.opt.save_every_e == 0:
                self.save(pjoin(self.opt.model_dir, 'E%04d.tar' % (epoch)), epoch, total_it=it)

            print('Validation time:')

            val_loss = 0
            val_accuracy = 0
            val_m2t_loss = 0
            val_m2t_accuracy = 0
            with torch.no_grad():
                for i, batch_data in enumerate(val_dataloader):
                    # self.
                    self.forward(batch_data, eval_mode=True)
                    val_loss += self.t2m_loss.item() / self.mot_n_word
                    val_accuracy += self.mot_n_correct / self.mot_n_word
                    if start_m2t:
                        self.forward_cycle(batch_data)
                        val_m2t_loss += self.m2t_loss.item() / self.txt_n_word
                        val_m2t_accuracy += self.txt_n_correct / self.txt_n_word
                    # val_loss_rec += self.l1_criterion(self.recon_motions, self.motions).item()
                    # val_loss_emb += self.embedding_loss.item()

            val_loss = val_loss / len(val_dataloader)
            val_accuracy = val_accuracy / len(val_dataloader)
            val_m2t_loss = val_m2t_loss / len(val_dataloader)
            val_m2t_accuracy = val_m2t_accuracy / len(val_dataloader)
            # val_loss = val_loss / (len(val_dataloader) + 1)
            # val_loss_rec = val_loss_rec / (len(val_dataloader) + 1)
            # val_loss_emb = val_loss_emb / (len(val_dataloader) + 1)
            print(self.mot_gold[0])
            print(self.mot_pred_seq.view(self.mot_gold.shape)[0])

            if start_m2t:
                gt_m2t_seq = ' '.join(w_vectorizer.itos(i) for i in self.txt_gold[0, :self.cap_lens[0]].cpu().numpy())
                pred_m2t_seq = ' '.join(
                    w_vectorizer.itos(i) for i in self.txt_pred_seq.view(self.txt_gold.shape)[0, :self.cap_lens[0]].cpu().numpy())
                print(gt_m2t_seq)
                print(pred_m2t_seq)

            print('Validation Loss: %.5f Validation Accuracy: %.4f Validation(M2T) Loss: %.5f Validation(M2T) Accuracy: %.4f ' %
                  (val_loss, val_accuracy, val_m2t_loss, val_m2t_accuracy))

            if val_loss < min_val_loss:
                min_val_loss = val_loss
                min_val_epoch = epoch
                self.save(pjoin(self.opt.model_dir, 'finest.tar'), epoch, it)
                print('Best Validation Model So Far!~')

            if epoch - min_val_epoch >= 5:
                print('Early Stopping!~')
                # break


class TransformerM2TJointTrainer(Trainer):
    def __init__(self, args, m2t_transformer, t2m_transformer):
        self.opt = args
        self.m2t_transformer = m2t_transformer
        self.t2m_transformer = t2m_transformer
        # self.quantizer = quantizer
        # self.vq_decoder = vq_decoder
        self.device = args.device

        # self.trg_pad_index = args.trg_pad_index
        # self.trg_start_index = args.trg_start_index
        # self.trg_end_index = args.trg_end_index
        # self.trg_num_vocab = args.trg_num_vocab

        if args.is_train:
            self.logger = Logger(args.log_dir)


    def forward(self, batch_data, data_aug=False):
        word_emb, word_tokens, caption, cap_lens, m_tokens, _ = batch_data
        word_emb = word_emb.detach().to(self.device).float()
        # pos_ohot = pos_ohot.detach().to(self.device).float()
        m_tokens = m_tokens.detach().to(self.device).long()
        word_tokens = word_tokens.detach().to(self.device).long()

        if data_aug:
            with torch.no_grad():
                pred_tokens, len_map = self.t2m_transformer.sample_batch(word_emb, cap_lens, trg_sos=self.opt.mot_start_idx,
                                                                         trg_eos=self.opt.mot_end_idx, max_steps=49,
                                                                         sample=True, top_k=100)

                for i in range(len(pred_tokens)):
                    pred_tokens[i, len_map[i] + 1:] = self.opt.mot_pad_idx
            m_tokens = pred_tokens

        cap_lens = cap_lens - 1
        self.cap_lens = cap_lens
        self.caption = caption

        self.gold = word_tokens[:, 1:]

        """Input pretrained word vector"""
        if self.opt.m2t_v3:
            trg_input = word_emb[:, :-1]
            self.trg_pred = self.m2t_transformer(m_tokens, trg_input, cap_lens)
        else:
            trg_input = word_tokens[:, :-1]
            self.trg_pred = self.m2t_transformer(m_tokens, trg_input)


    def backward(self):
        # print(self.trg_pred.shape, self.gold.shape)
        trg_pred = self.trg_pred.view(-1, self.trg_pred.shape[-1]).clone()
        # print(trg_pred[0])
        gold = self.gold.contiguous().view(-1).clone()
        self.loss, self.pred_seq, self.n_correct, self.n_word = cal_performance(trg_pred, gold, self.opt.txt_pad_idx,
                                                            smoothing=self.opt.label_smoothing)
        # print(gold, self.pred_seq)
        # self.loss = loss / n_word
        loss_logs = OrderedDict({})
        loss_logs['loss'] = self.loss.item() / self.n_word
        loss_logs['accuracy'] = self.n_correct / self.n_word

        return loss_logs

    def update(self):
        self.zero_grad([self.m2t_transformer])
        # time2_0 = time.time()
        # print("\t\t Zero Grad:%5f" % (time2_0 - time1))
        loss_logs = self.backward()
        self.loss.backward()

        # time2_3 = time.time()
        # print("\t\t Clip Norm :%5f" % (time2_3 - time2_2))
        self.step([self.opt_m2t_transformer])

        return loss_logs

    def save(self, file_name, ep, total_it):

        state = {
            'm2t_transformer': self.m2t_transformer.state_dict(),

            'opt_m2t_transformer': self.opt_m2t_transformer.state_dict(),

            'ep': ep,
            'total_it': total_it,
        }
        torch.save(state, file_name)

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        self.m2t_transformer.load_state_dict(checkpoint['m2t_transformer'])

        self.opt_m2t_transformer.load_state_dict(checkpoint['opt_m2t_transformer'])
        # if self.opt.use_gan:
        #     self.discriminator.load_state_dict(checkpoint['discriminator'])
        #     self.opt_discriminator.load_state_dict(checkpoint['opt_discriminator'])
        return checkpoint['ep'], checkpoint['total_it']


    def train(self, train_dataloader, val_dataloader, w_vectorizer):
        self.m2t_transformer.to(self.device)
        self.t2m_transformer.to(self.device)
        # self.vq_decoder.to(self.device)
        # self.quantizer.to(self.device)

        self.opt_m2t_transformer = optim.Adam(self.m2t_transformer.parameters(), lr=self.opt.lr)


        epoch = 0
        it = 0
        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')
            epoch, it = self.resume(model_dir)

        start_time = time.time()
        total_iters = self.opt.max_epoch * len(train_dataloader)
        print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_dataloader), len(val_dataloader)))
        val_loss = 0
        val_accuracy = 0
        min_val_loss = np.inf
        min_val_epoch = epoch
        logs = OrderedDict()
        while epoch < self.opt.max_epoch:
            for i, batch_data in enumerate(train_dataloader):
                self.m2t_transformer.train()
                self.t2m_transformer.eval()

                coin = np.random.choice([True, False, False, False, False])

                self.forward(batch_data, coin)

                log_dict = self.update()
                # continue
                # time3 = time.time()
                # print('Update Time: %.5f s' % (time3 - time2))
                # time0 = time3
                for k, v in log_dict.items():
                    if k not in logs:
                        logs[k] = v
                    else:
                        logs[k] += v

                it += 1
                if it % self.opt.log_every == 0:
                    mean_loss = OrderedDict({'val_loss': val_loss, 'val_accuracy':val_accuracy})
                    self.logger.scalar_summary('val_loss', val_loss, it)
                    self.logger.scalar_summary('val_accuracy', val_accuracy, it)

                    for tag, value in logs.items():
                        self.logger.scalar_summary(tag, value / self.opt.log_every, it)
                        mean_loss[tag] = value / self.opt.log_every
                    logs = OrderedDict()
                    print_current_loss(start_time, it, total_iters, mean_loss, epoch, i)

                if it % self.opt.save_latest == 0:
                    self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            epoch += 1
            if epoch % self.opt.save_every_e == 0:
                self.save(pjoin(self.opt.model_dir, 'E%04d.tar' % (epoch)), epoch, total_it=it)

            print('Validation time:')

            val_loss = 0
            val_accuracy = 0
            with torch.no_grad():
                for i, batch_data in enumerate(val_dataloader):
                    self.forward(batch_data)
                    self.backward()
                    val_loss += self.loss.item() / self.n_word
                    val_accuracy += self.n_correct / self.n_word
                    # val_loss_rec += self.l1_criterion(self.recon_motions, self.motions).item()
                    # val_loss_emb += self.embedding_loss.item()

            val_loss = val_loss / len(val_dataloader)
            val_accuracy = val_accuracy / len(val_dataloader)
            # val_loss = val_loss / (len(val_dataloader) + 1)
            # val_loss_rec = val_loss_rec / (len(val_dataloader) + 1)
            # val_loss_emb = val_loss_emb / (len(val_dataloader) + 1)

            print('Validation Loss: %.5f Validation Accuracy: %.4f' % (val_loss, val_accuracy))

            if val_loss < min_val_loss:
                min_val_loss = val_loss
                min_val_epoch = epoch
                self.save(pjoin(self.opt.model_dir, 'finest.tar'), epoch, it)
                print('Best Validation Model So Far!~')

            print(self.gold[0])
            print(self.pred_seq.view(self.gold.shape)[0])
            gt_seq = ' '.join(w_vectorizer.itos(i) for i in self.gold[0, :self.cap_lens[0]].cpu().numpy())
            pred_seq = ' '.join(
                w_vectorizer.itos(i) for i in self.pred_seq.view(self.gold.shape)[0, :self.cap_lens[0]].cpu().numpy())
            print(gt_seq)
            print(pred_seq)
            # if epoch % self.opt.eval_every_e == 0:
                # self.quantizer.eval()
                # self.vq_decoder.eval()
                # with torch.no_grad():
                #     pred_seq = self.pred_seq.view(self.gold.shape)[0:1]
                #     # print(pred_seq.shape)
                #     non_pad_mask = self.gold[0:1].ne(self.opt.trg_pad_idx)
                #     pred_seq = pred_seq.masked_select(non_pad_mask).unsqueeze(0)
                #     # print(non_pad_mask.shape)
                #     # print(pred_seq.shape)
                #     # print(pred_seq)
                #     # print(self.gold[0:1])
                #     vq_latent = self.quantizer.get_codebook_entry(pred_seq)
                #     # print(vq_latent.shape)
                #
                #     rec_motion = self.vq_decoder(vq_latent)
                #
                # save_dir = pjoin(self.opt.eval_dir, 'E%04d' % (epoch))
                # os.makedirs(save_dir, exist_ok=True)
                # plot_eval(rec_motion.detach().cpu().numpy(), self.caption[0:1], save_dir)
                # save_dir = pjoin(self.opt.eval_dir, 'E%04d' % epoch)
                # os.makedirs()

            if epoch - min_val_epoch >= 5:
                print('Early Stopping!~')
                # break


# Motion to text generation without tokenization
class TransformerM2T_NoT_Trainer(Trainer):
    def __init__(self, args, m2t_transformer):
        self.opt = args
        self.m2t_transformer = m2t_transformer
        # self.quantizer = quantizer
        # self.vq_decoder = vq_decoder
        self.device = args.device

        if args.is_train:
            self.logger = Logger(args.log_dir)

    def forward(self, batch_data):
        _, word_tokens, caption, cap_lens, motion, m_length = batch_data
        # word_emb = word_emb.detach().to(self.device).float()
        # pos_ohot = pos_ohot.detach().to(self.device).float()
        motion = motion.detach().to(self.device).float()
        word_tokens = word_tokens.detach().to(self.device).long()

        cap_lens = cap_lens - 1
        self.cap_lens = cap_lens
        self.caption = caption

        self.gold = word_tokens[:, 1:]

        trg_input = word_tokens[:, :-1]

        self.trg_pred = self.m2t_transformer(motion, trg_input, m_length)


        # one_hot_indices = F.one_hot(encoding_indices, num_classes=self.args.trg_num_vocab)

    def backward(self):
        # print(self.trg_pred.shape, self.gold.shape)
        trg_pred = self.trg_pred.view(-1, self.trg_pred.shape[-1]).clone()
        # print(trg_pred[0])
        gold = self.gold.contiguous().view(-1).clone()
        self.loss, self.pred_seq, self.n_correct, self.n_word = cal_performance(trg_pred, gold, self.opt.txt_pad_idx,
                                                            smoothing=self.opt.label_smoothing)
        # print(gold, self.pred_seq)
        # self.loss = loss / n_word
        loss_logs = OrderedDict({})
        loss_logs['loss'] = self.loss.item() / self.n_word
        loss_logs['accuracy'] = self.n_correct / self.n_word

        return loss_logs

    def update(self):
        self.zero_grad([self.opt_m2t_transformer])
        # time2_0 = time.time()
        # print("\t\t Zero Grad:%5f" % (time2_0 - time1))
        loss_logs = self.backward()
        self.loss.backward()

        # time2_3 = time.time()
        # print("\t\t Clip Norm :%5f" % (time2_3 - time2_2))
        self.step([self.opt_m2t_transformer])

        return loss_logs

    def save(self, file_name, ep, total_it):

        state = {
            'm2t_transformer': self.m2t_transformer.state_dict(),

            'opt_m2t_transformer': self.opt_m2t_transformer.state_dict(),

            'ep': ep,
            'total_it': total_it,
        }
        torch.save(state, file_name)

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        self.m2t_transformer.load_state_dict(checkpoint['m2t_transformer'])

        self.opt_m2t_transformer.load_state_dict(checkpoint['opt_m2t_transformer'])
        # if self.opt.use_gan:
        #     self.discriminator.load_state_dict(checkpoint['discriminator'])
        #     self.opt_discriminator.load_state_dict(checkpoint['opt_discriminator'])
        return checkpoint['ep'], checkpoint['total_it']


    def train(self, train_dataloader, val_dataloader, w_vectorizer):
        self.m2t_transformer.to(self.device)
        # self.vq_decoder.to(self.device)
        # self.quantizer.to(self.device)

        self.opt_m2t_transformer = optim.Adam(self.m2t_transformer.parameters(), lr=self.opt.lr)


        epoch = 0
        it = 0
        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')
            epoch, it = self.resume(model_dir)

        start_time = time.time()
        total_iters = self.opt.max_epoch * len(train_dataloader)
        print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_dataloader), len(val_dataloader)))
        val_loss = 0
        val_accuracy = 0
        min_val_loss = np.inf
        min_val_epoch = epoch
        logs = OrderedDict()
        while epoch < self.opt.max_epoch:
            for i, batch_data in enumerate(train_dataloader):
                self.m2t_transformer.train()

                self.forward(batch_data)

                log_dict = self.update()
                # continue
                # time3 = time.time()
                # print('Update Time: %.5f s' % (time3 - time2))
                # time0 = time3
                for k, v in log_dict.items():
                    if k not in logs:
                        logs[k] = v
                    else:
                        logs[k] += v

                it += 1
                if it % self.opt.log_every == 0:
                    mean_loss = OrderedDict({'val_loss': val_loss, 'val_accuracy':val_accuracy})
                    self.logger.scalar_summary('val_loss', val_loss, it)
                    self.logger.scalar_summary('val_accuracy', val_accuracy, it)

                    for tag, value in logs.items():
                        self.logger.scalar_summary(tag, value / self.opt.log_every, it)
                        mean_loss[tag] = value / self.opt.log_every
                    logs = OrderedDict()
                    print_current_loss(start_time, it, total_iters, mean_loss, epoch, i)

                if it % self.opt.save_latest == 0:
                    self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            epoch += 1
            if epoch % self.opt.save_every_e == 0:
                self.save(pjoin(self.opt.model_dir, 'E%04d.tar' % (epoch)), epoch, total_it=it)

            print('Validation time:')

            val_loss = 0
            val_accuracy = 0
            with torch.no_grad():
                for i, batch_data in enumerate(val_dataloader):
                    self.forward(batch_data)
                    self.backward()
                    val_loss += self.loss.item() / self.n_word
                    val_accuracy += self.n_correct / self.n_word
                    # val_loss_rec += self.l1_criterion(self.recon_motions, self.motions).item()
                    # val_loss_emb += self.embedding_loss.item()

            val_loss = val_loss / len(val_dataloader)
            val_accuracy = val_accuracy / len(val_dataloader)
            # val_loss = val_loss / (len(val_dataloader) + 1)
            # val_loss_rec = val_loss_rec / (len(val_dataloader) + 1)
            # val_loss_emb = val_loss_emb / (len(val_dataloader) + 1)

            print('Validation Loss: %.5f Validation Accuracy: %.4f' % (val_loss, val_accuracy))

            if val_loss < min_val_loss:
                min_val_loss = val_loss
                min_val_epoch = epoch
                self.save(pjoin(self.opt.model_dir, 'finest.tar'), epoch, it)
                print('Best Validation Model So Far!~')

            print(self.gold[0])
            print(self.pred_seq.view(self.gold.shape)[0])
            gt_seq = ' '.join(w_vectorizer.itos(i) for i in self.gold[0, :self.cap_lens[0]].cpu().numpy())
            pred_seq = ' '.join(
                w_vectorizer.itos(i) for i in self.pred_seq.view(self.gold.shape)[0, :self.cap_lens[0]].cpu().numpy())
            print(gt_seq)
            print(pred_seq)
            # if epoch % self.opt.eval_every_e == 0:
                # self.quantizer.eval()
                # self.vq_decoder.eval()
                # with torch.no_grad():
                #     pred_seq = self.pred_seq.view(self.gold.shape)[0:1]
                #     # print(pred_seq.shape)
                #     non_pad_mask = self.gold[0:1].ne(self.opt.trg_pad_idx)
                #     pred_seq = pred_seq.masked_select(non_pad_mask).unsqueeze(0)
                #     # print(non_pad_mask.shape)
                #     # print(pred_seq.shape)
                #     # print(pred_seq)
                #     # print(self.gold[0:1])
                #     vq_latent = self.quantizer.get_codebook_entry(pred_seq)
                #     # print(vq_latent.shape)
                #
                #     rec_motion = self.vq_decoder(vq_latent)
                #
                # save_dir = pjoin(self.opt.eval_dir, 'E%04d' % (epoch))
                # os.makedirs(save_dir, exist_ok=True)
                # plot_eval(rec_motion.detach().cpu().numpy(), self.caption[0:1], save_dir)
                # save_dir = pjoin(self.opt.eval_dir, 'E%04d' % epoch)
                # os.makedirs()

            if epoch - min_val_epoch >= 5:
                print('Early Stopping!~')
                # break


class M2T_Seq2Seq_Trainer(Trainer):
    def __init__(self, args, model):
        self.opt = args
        self.model = model
        # self.quantizer = quantizer
        # self.vq_decoder = vq_decoder
        self.device = args.device

        if args.is_train:
            self.logger = Logger(args.log_dir)

    def forward(self, batch_data):
        _, word_tokens, caption, cap_lens, motion, m_length = batch_data
        # word_emb = word_emb.detach().to(self.device).float()
        # pos_ohot = pos_ohot.detach().to(self.device).float()
        # print(motion.shape)
        # print(m_length)
        motion = motion.detach().to(self.device).float()
        word_tokens = word_tokens.detach().to(self.device).long()

        cap_lens = cap_lens - 1
        self.cap_lens = cap_lens
        self.caption = caption

        self.gold = word_tokens[:, 1:]

        trg_input = word_tokens[:, :-1]
        self.trg_pred = self.model(motion, trg_input, m_length, self.opt.tf_ratio)

        # one_hot_indices = F.one_hot(encoding_indices, num_classes=self.args.trg_num_vocab)

    def backward(self):
        # print(self.trg_pred.shape, self.gold.shape)
        trg_pred = self.trg_pred.view(-1, self.trg_pred.shape[-1]).clone()
        # print(trg_pred[0])
        gold = self.gold.contiguous().view(-1).clone()
        self.loss, self.pred_seq, self.n_correct, self.n_word = cal_performance(trg_pred, gold, self.opt.txt_pad_idx,
                                                            smoothing=self.opt.label_smoothing)
        # print(gold, self.pred_seq)
        # self.loss = loss / n_word
        loss_logs = OrderedDict({})
        loss_logs['loss'] = self.loss.item() / self.n_word
        loss_logs['accuracy'] = self.n_correct / self.n_word

        return loss_logs

    def update(self):
        self.zero_grad([self.opt_model])
        # time2_0 = time.time()
        # print("\t\t Zero Grad:%5f" % (time2_0 - time1))
        loss_logs = self.backward()
        self.loss.backward()

        # time2_3 = time.time()
        # print("\t\t Clip Norm :%5f" % (time2_3 - time2_2))
        self.step([self.opt_model])

        return loss_logs

    def save(self, file_name, ep, total_it):

        state = {
            'model': self.model.state_dict(),

            'opt_model': self.opt_model.state_dict(),

            'ep': ep,
            'total_it': total_it,
        }
        torch.save(state, file_name)

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])

        self.opt_model.load_state_dict(checkpoint['opt_model'])
        # if self.opt.use_gan:
        #     self.discriminator.load_state_dict(checkpoint['discriminator'])
        #     self.opt_discriminator.load_state_dict(checkpoint['opt_discriminator'])
        return checkpoint['ep'], checkpoint['total_it']


    def train(self, train_dataloader, val_dataloader, w_vectorizer):
        self.model.to(self.device)
        # self.vq_decoder.to(self.device)
        # self.quantizer.to(self.device)

        self.opt_model = optim.Adam(self.model.parameters(), lr=self.opt.lr)


        epoch = 0
        it = 0
        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')
            epoch, it = self.resume(model_dir)

        start_time = time.time()
        total_iters = self.opt.max_epoch * len(train_dataloader)
        print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_dataloader), len(val_dataloader)))
        val_loss = 0
        val_accuracy = 0
        min_val_loss = np.inf
        min_val_epoch = epoch
        logs = OrderedDict()
        while epoch < self.opt.max_epoch:
            for i, batch_data in enumerate(train_dataloader):
                self.model.train()

                self.forward(batch_data)

                log_dict = self.update()
                # continue
                # time3 = time.time()
                # print('Update Time: %.5f s' % (time3 - time2))
                # time0 = time3
                for k, v in log_dict.items():
                    if k not in logs:
                        logs[k] = v
                    else:
                        logs[k] += v

                it += 1
                if it % self.opt.log_every == 0:
                    mean_loss = OrderedDict({'val_loss': val_loss, 'val_accuracy':val_accuracy})
                    self.logger.scalar_summary('val_loss', val_loss, it)
                    self.logger.scalar_summary('val_accuracy', val_accuracy, it)

                    for tag, value in logs.items():
                        self.logger.scalar_summary(tag, value / self.opt.log_every, it)
                        mean_loss[tag] = value / self.opt.log_every
                    logs = OrderedDict()
                    print_current_loss(start_time, it, total_iters, mean_loss, epoch, i)

                if it % self.opt.save_latest == 0:
                    self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            epoch += 1
            if epoch % self.opt.save_every_e == 0:
                self.save(pjoin(self.opt.model_dir, 'E%04d.tar' % (epoch)), epoch, total_it=it)

            print('Validation time:')

            val_loss = 0
            val_accuracy = 0
            with torch.no_grad():
                for i, batch_data in enumerate(val_dataloader):
                    self.forward(batch_data)
                    self.backward()
                    val_loss += self.loss.item() / self.n_word
                    val_accuracy += self.n_correct / self.n_word
                    # val_loss_rec += self.l1_criterion(self.recon_motions, self.motions).item()
                    # val_loss_emb += self.embedding_loss.item()

            val_loss = val_loss / len(val_dataloader)
            val_accuracy = val_accuracy / len(val_dataloader)
            # val_loss = val_loss / (len(val_dataloader) + 1)
            # val_loss_rec = val_loss_rec / (len(val_dataloader) + 1)
            # val_loss_emb = val_loss_emb / (len(val_dataloader) + 1)

            print('Validation Loss: %.5f Validation Accuracy: %.4f' % (val_loss, val_accuracy))

            if val_loss < min_val_loss:
                min_val_loss = val_loss
                min_val_epoch = epoch
                self.save(pjoin(self.opt.model_dir, 'finest.tar'), epoch, it)
                print('Best Validation Model So Far!~')

            print(self.gold[0])
            print(self.pred_seq.view(self.gold.shape)[0])
            gt_seq = ' '.join(w_vectorizer.itos(i) for i in self.gold[0, :self.cap_lens[0]].cpu().numpy())
            pred_seq = ' '.join(
                w_vectorizer.itos(i) for i in self.pred_seq.view(self.gold.shape)[0, :self.cap_lens[0]].cpu().numpy())
            print(gt_seq)
            print(pred_seq)
            # if epoch % self.opt.eval_every_e == 0:
                # self.quantizer.eval()
                # self.vq_decoder.eval()
                # with torch.no_grad():
                #     pred_seq = self.pred_seq.view(self.gold.shape)[0:1]
                #     # print(pred_seq.shape)
                #     non_pad_mask = self.gold[0:1].ne(self.opt.trg_pad_idx)
                #     pred_seq = pred_seq.masked_select(non_pad_mask).unsqueeze(0)
                #     # print(non_pad_mask.shape)
                #     # print(pred_seq.shape)
                #     # print(pred_seq)
                #     # print(self.gold[0:1])
                #     vq_latent = self.quantizer.get_codebook_entry(pred_seq)
                #     # print(vq_latent.shape)
                #
                #     rec_motion = self.vq_decoder(vq_latent)
                #
                # save_dir = pjoin(self.opt.eval_dir, 'E%04d' % (epoch))
                # os.makedirs(save_dir, exist_ok=True)
                # plot_eval(rec_motion.detach().cpu().numpy(), self.caption[0:1], save_dir)
                # save_dir = pjoin(self.opt.eval_dir, 'E%04d' % epoch)
                # os.makedirs()

            if epoch - min_val_epoch >= 5:
                print('Early Stopping!~')
                # break


class M2T_GAN_Trainer(Trainer):
    def __init__(self, args, discriminator, generator):
        self.opt = args
        self.model = generator
        self.discriminator = discriminator
        # self.quantizer = quantizer
        # self.vq_decoder = vq_decoder
        self.device = args.device

        if args.is_train:
            self.logger = Logger(args.log_dir)
            self.gan_criterion = torch.nn.BCEWithLogitsLoss()


    def forward(self, batch_data):
        _, word_tokens, caption, cap_lens, motion, m_length = batch_data
        # word_emb = word_emb.detach().to(self.device).float()
        # pos_ohot = pos_ohot.detach().to(self.device).float()
        # print(motion.shape)
        # print(m_length)
        motion = motion.detach().to(self.device).float()
        word_tokens = word_tokens.detach().to(self.device).long()

        cap_lens = cap_lens - 1
        self.cap_lens = cap_lens
        self.caption = caption

        self.gold = word_tokens[:, 1:]

        trg_input = word_tokens[:, :-1]
        self.trg_pred = self.model(motion, trg_input, m_length, self.opt.tf_ratio)

        # one_hot_indices = F.one_hot(encoding_indices, num_classes=self.args.trg_num_vocab)

    def backward_D(self):
        self.real_words = F.one_hot(self.gold, num_classes=self.opt.n_txt_vocab).float()
        self.fake_words = F.gumbel_softmax(self.trg_pred, tau=1, hard=True, dim=-1)
        real_labels = self.discriminator(self.real_words.detach())
        fake_labels = self.discriminator(self.fake_words.detach())

        ones = self.ones_like(real_labels)
        zeros = self.zeros_like(fake_labels)
        self.loss_D_T = self.gan_criterion(real_labels, ones)
        self.loss_D_F = self.gan_criterion(fake_labels, zeros)
        self.loss_D = self.loss_D_T + self.loss_D_F
        self.acc_D_F = (fake_labels < 0.5).sum().item() / len(ones)
        self.acc_D_T = (real_labels > 0.5).sum().item() / len(zeros)

    def backward_G(self):
        self.fake_words = F.gumbel_softmax(self.trg_pred, tau=1, hard=True, dim=-1)
        fake_labels = self.discriminator(self.fake_words)

        ones = self.ones_like(fake_labels)
        self.loss_G = self.gan_criterion(fake_labels, ones)
        self.acc_G = (fake_labels > 0.5).sum().item() / len(ones)


    def update(self):
        loss_logs = OrderedDict({})

        self.zero_grad([self.opt_discriminator])
        self.backward_D()
        self.loss_D.backward(retain_graph=self.opt.start_gan)
        self.step([self.opt_discriminator])

        loss_logs['loss_D_F'] = self.loss_D_F.item()
        loss_logs['loss_D_T'] = self.loss_D_T.item()
        loss_logs['acc_D_F'] = self.acc_D_F
        loss_logs['acc_D_T'] = self.acc_D_T
        loss_logs['loss_D'] = self.loss_D.item()
        if self.opt.start_gan:
            self.zero_grad([self.opt_model])
            self.backward_G()
            self.loss_G.backward()
            self.clip_norm([self.model])
            self.step([self.opt_model])
            loss_logs['loss_G'] = self.loss_G.item()
            loss_logs['acc_G'] = self.acc_G
        return loss_logs

    def save(self, file_name, ep, total_it):

        state = {
            'model': self.model.state_dict(),

            'opt_model': self.opt_model.state_dict(),

            'discriminator': self.discriminator.state_dict(),

            'opt_discriminator': self.opt_discriminator.state_dict(),

            'ep': ep,
            'total_it': total_it,
        }
        torch.save(state, file_name)

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])

        self.opt_model.load_state_dict(checkpoint['opt_model'])
        self.opt_discriminator.load_state_dict(checkpoint['opt_discriminator'])
        # if self.opt.use_gan:
        #     self.discriminator.load_state_dict(checkpoint['discriminator'])
        #     self.opt_discriminator.load_state_dict(checkpoint['opt_discriminator'])
        return checkpoint['ep'], checkpoint['total_it']


    def train(self, train_dataloader, val_dataloader, w_vectorizer):
        self.model.to(self.device)
        self.discriminator.to(self.device)
        # self.vq_decoder.to(self.device)
        # self.quantizer.to(self.device)

        self.opt_model = optim.Adam(self.model.parameters(), lr=self.opt.lr)
        self.opt_discriminator = optim.Adam(self.discriminator.parameters(), lr=self.opt.lr*0.01)


        epoch = 0
        it = 0
        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')
            epoch, it = self.resume(model_dir)

        start_time = time.time()
        total_iters = self.opt.max_epoch * len(train_dataloader)
        print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_dataloader), len(val_dataloader)))
        val_loss = 0
        val_accuracy = 0
        min_val_loss = np.inf
        min_val_epoch = epoch
        logs = OrderedDict()
        while epoch < self.opt.max_epoch:
            for i, batch_data in enumerate(train_dataloader):
                self.model.train()

                self.forward(batch_data)

                log_dict = self.update()
                # continue
                # time3 = time.time()
                # print('Update Time: %.5f s' % (time3 - time2))
                # time0 = time3
                for k, v in log_dict.items():
                    if k not in logs:
                        logs[k] = v
                    else:
                        logs[k] += v

                it += 1
                if it % self.opt.log_every == 0:
                    mean_loss = OrderedDict({'val_loss': val_loss})
                    self.logger.scalar_summary('val_loss', val_loss, it)
                    # self.logger.scalar_summary('val_accuracy', val_accuracy, it)

                    for tag, value in logs.items():
                        self.logger.scalar_summary(tag, value / self.opt.log_every, it)
                        mean_loss[tag] = value / self.opt.log_every
                    logs = OrderedDict()
                    print_current_loss(start_time, it, total_iters, mean_loss, epoch, i)

                if it % self.opt.save_latest == 0:
                    self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            epoch += 1
            if epoch % self.opt.save_every_e == 0:
                self.save(pjoin(self.opt.model_dir, 'E%04d.tar' % (epoch)), epoch, total_it=it)

            print('Validation time:')

            val_loss = 0
            val_D = 0
            val_G = 0
            val_acc_D = 0
            val_acc_G = 0
            with torch.no_grad():
                for i, batch_data in enumerate(val_dataloader):
                    self.forward(batch_data)
                    self.backward_D()
                    self.backward_G()
                    val_D += self.loss_D.item()
                    val_G += self.loss_G.item()

                    val_acc_D += (self.acc_D_T + self.acc_D_F)/2
                    val_acc_G += self.acc_G

            val_D /= len(val_dataloader)
            val_G /= len(val_dataloader)
            val_acc_D /= len(val_dataloader)
            val_acc_G /= len(val_dataloader)

            if self.opt.start_gan:
                val_loss = val_G
            else:
                val_loss = val_D

            print('Val_D Loss: %.5f Val_D Accuracy: %.4f '
                  'Val_G Loss: %.5f Val_G Accuracy: %.4f' % (val_D, val_acc_D, val_G, val_acc_G))

            if val_loss < min_val_loss:
                min_val_loss = val_loss
                min_val_epoch = epoch
                self.save(pjoin(self.opt.model_dir, 'finest.tar'), epoch, it)
                print('Best Validation Model So Far!~')

            # print(self.gold[0])
            # pred_seq =  self.trg_pred.max(1)[1]
            # print(pred_seq.view(self.gold.shape)[0])
            # gt_seq = ' '.join(w_vectorizer.itos(i) for i in self.gold[0, :self.cap_lens[0]].cpu().numpy())
            # pred_seq = ' '.join(
            #     w_vectorizer.itos(i) for i in pred_seq.view(self.gold.shape)[0, :self.cap_lens[0]].cpu().numpy())
            # print(gt_seq)
            # print(pred_seq)
            # if epoch % self.opt.eval_every_e == 0:
                # self.quantizer.eval()
                # self.vq_decoder.eval()
                # with torch.no_grad():
                #     pred_seq = self.pred_seq.view(self.gold.shape)[0:1]
                #     # print(pred_seq.shape)
                #     non_pad_mask = self.gold[0:1].ne(self.opt.trg_pad_idx)
                #     pred_seq = pred_seq.masked_select(non_pad_mask).unsqueeze(0)
                #     # print(non_pad_mask.shape)
                #     # print(pred_seq.shape)
                #     # print(pred_seq)
                #     # print(self.gold[0:1])
                #     vq_latent = self.quantizer.get_codebook_entry(pred_seq)
                #     # print(vq_latent.shape)
                #
                #     rec_motion = self.vq_decoder(vq_latent)
                #
                # save_dir = pjoin(self.opt.eval_dir, 'E%04d' % (epoch))
                # os.makedirs(save_dir, exist_ok=True)
                # plot_eval(rec_motion.detach().cpu().numpy(), self.caption[0:1], save_dir)
                # save_dir = pjoin(self.opt.eval_dir, 'E%04d' % epoch)
                # os.makedirs()

            if epoch - min_val_epoch >= 5:
                print('Early Stopping!~')
                # break
