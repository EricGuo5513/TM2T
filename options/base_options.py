import argparse
import os
import torch


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--name', type=str, default="test", help='Name of this trial')
        self.parser.add_argument('--tokenizer_name', type=str, default="VQVAEV3_CB1024_CMT_H1024_NRES3", help='Name of this trial')

        self.parser.add_argument("--gpu_id", type=int, default=-1, help='GPU id')

        self.parser.add_argument('--dataset_name', type=str, default='t2m', help='Dataset Name')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')

        self.parser.add_argument("--unit_length", type=int, default=4, help="Length of motion")
        self.parser.add_argument("--max_text_len", type=int, default=20, help="Length of motion")
        # self.parser.add_argument("--max_motion_len", type=int, default=55, help="Length of motion")

        self.parser.add_argument('--d_model', type=int, default=512, help='Dimension of hidden unit in GRU')
        self.parser.add_argument('--d_inner_hid', type=int, default=2048, help='Dimension of hidden unit in GRU')
        self.parser.add_argument('--d_k', type=int, default=64, help='Dimension of hidden unit in GRU')
        self.parser.add_argument('--d_v', type=int, default=64, help='Dimension of hidden unit in GRU')

        self.parser.add_argument('--n_head', type=int, default=8, help='Dimension of hidden unit in GRU')
        self.parser.add_argument('--n_enc_layers', type=int, default=6, help='Dimension of hidden unit in GRU')
        self.parser.add_argument('--n_dec_layers', type=int, default=6, help='Dimension of hidden unit in GRU')

        self.parser.add_argument('--dropout', type=float, default=0.1, help='Dimension of hidden unit in GRU')

        self.parser.add_argument('--proj_share_weight', action="store_true", help='Training iterations')

        self.parser.add_argument('--t2m_v2', action="store_true", help='Training iterations')
        self.parser.add_argument('--m2t_v3', action="store_true", help='Training iterations')

        # Hyper-Parameters for Seq2Seq Motion2text
        self.parser.add_argument('--dim_mot_hid', type=int, default=1024, help='Dimension of hidden unit in GRU')
        self.parser.add_argument('--dim_txt_hid', type=int, default=512, help='Dimension of hidden unit in GRU')
        self.parser.add_argument('--n_mot_layers', type=int, default=1, help='Dimension of hidden unit in GRU')
        self.parser.add_argument('--early_or_late', type=str, default="early", help='Dimension of hidden unit in GRU')


        # Hyper-Parameters for tokenizer
        self.parser.add_argument('--q_mode', type=str, default='cmt', help='Dataset Name')
        self.parser.add_argument('--dim_vq_enc_hidden', type=int, default=1024, help='Dimension of hidden unit in GRU')
        self.parser.add_argument('--dim_vq_dec_hidden', type=int, default=1024, help='Dimension of hidden unit in GRU')
        self.parser.add_argument('--dim_vq_latent', type=int, default=1024, help='Dimension of hidden unit in GRU')

        self.parser.add_argument('--lambda_beta', type=float, default=1, help='Layers of GRU')

        self.parser.add_argument('--n_down', type=int, default=2, help='Dimension of hidden unit in GRU')
        self.parser.add_argument('--n_resblk', type=int, default=3, help='Dimension of hidden unit in GRU')
        self.parser.add_argument('--codebook_size', type=int, default=1024, help='Dimension of hidden unit in GRU')

        self.initialized = True



    def parse(self):
        if not self.initialized:
            self.initialize()

        self.opt = self.parser.parse_args()

        self.opt.is_train = self.is_train

        if self.opt.gpu_id != -1:
            # self.opt.gpu_id = int(self.opt.gpu_id)
            torch.cuda.set_device(self.opt.gpu_id)

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')
        if self.is_train:
            # save to the disk
            expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.dataset_name, self.opt.name)
            if not os.path.exists(expr_dir):
                os.makedirs(expr_dir)
            file_name = os.path.join(expr_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')
        return self.opt