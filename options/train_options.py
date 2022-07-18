from options.base_options import BaseOptions
import argparse

class TrainT2MOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
        self.parser.add_argument('--max_epoch', type=int, default=300, help='Training iterations')
        self.parser.add_argument('--lambda_m2t', type=float, default=1.0, help='Training iterations')

        self.parser.add_argument('--start_m2t_ep', type=int, default=100, help='Dimension of hidden unit in GRU')
        self.parser.add_argument('--top_k', type=int, default=-1, help='Dimension of hidden unit in GRU')

        self.parser.add_argument('--tf_ratio', type=float, default=0.4, help='Training iterations')

        self.parser.add_argument('--is_continue', action="store_true", help='Training iterations')
        self.parser.add_argument('--text_aug', action="store_true", help='Training iterations')

        self.parser.add_argument('--label_smoothing', action='store_true')

        self.parser.add_argument('--lr', type=float, default=2e-4, help='Layers of GRU')
        self.parser.add_argument('--log_every', type=int, default=50, help='Frequency of printing training progress')
        self.parser.add_argument('--save_every_e', type=int, default=10, help='Frequency of printing training progress')
        self.parser.add_argument('--eval_every_e', type=int, default=5, help='Frequency of printing training progress')
        self.parser.add_argument('--save_latest', type=int, default=500, help='Frequency of printing training progress')

        self.is_train = True



class TrainVQTokenizerOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.parser.add_argument('--name', type=str, default="test", help='Name of this trial')
        self.parser.add_argument('--decomp_name', type=str, default="Decomp_SP001_SM001_H512", help='Name of this trial')

        self.parser.add_argument("--gpu_id", type=int, default=-1,
                                 help='GPU id')

        self.parser.add_argument('--dataset_name', type=str, default='t2m', help='Dataset Name')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')

        self.parser.add_argument("--window_size", type=int, default=64, help="Length of motion")

        self.parser.add_argument('--q_mode', type=str, default='cmt', help='Dataset Name')
        self.parser.add_argument('--dim_vq_enc_hidden', type=int, default=1024, help='Dimension of hidden unit in GRU')
        self.parser.add_argument('--dim_vq_dec_hidden', type=int, default=1024, help='Dimension of hidden unit in GRU')
        self.parser.add_argument('--dim_vq_latent', type=int, default=1024, help='Dimension of hidden unit in GRU')
        self.parser.add_argument('--dim_vq_dis_hidden', type=int, default=512, help='Dimension of hidden unit in GRU')

        self.parser.add_argument('--n_layers_dis', type=int, default=2, help='Dimension of hidden unit in GRU')
        self.parser.add_argument('--n_down', type=int, default=2, help='Dimension of hidden unit in GRU')
        self.parser.add_argument('--n_resblk', type=int, default=2, help='Dimension of hidden unit in GRU')
        self.parser.add_argument('--codebook_size', type=int, default=1024, help='Dimension of hidden unit in GRU')

        self.parser.add_argument('--use_gan', action="store_true", help='Training iterations')
        self.parser.add_argument('--use_feat_M', action="store_true", help='Training iterations')
        self.parser.add_argument('--use_percep', action="store_true", help='Training iterations')


        self.parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
        self.parser.add_argument('--max_epoch', type=int, default=300, help='Training iterations')

        self.parser.add_argument('--feat_bias', type=float, default=5, help='Layers of GRU')

        self.parser.add_argument('--start_dis_epoch', type=float, default=10, help='Layers of GRU')

        self.parser.add_argument('--lambda_adv', type=float, default=0.5, help='Layers of GRU')
        self.parser.add_argument('--lambda_fm', type=float, default=0.01, help='Layers of GRU')
        self.parser.add_argument('--lambda_beta', type=float, default=1, help='Layers of GRU')


        self.parser.add_argument('--lr', type=float, default=1e-4, help='Layers of GRU')

        self.parser.add_argument('--is_continue', action="store_true", help='Training iterations')

        self.parser.add_argument('--log_every', type=int, default=50, help='Frequency of printing training progress')
        self.parser.add_argument('--save_every_e', type=int, default=10, help='Frequency of printing training progress')
        self.parser.add_argument('--eval_every_e', type=int, default=3, help='Frequency of printing training progress')
        self.parser.add_argument('--save_latest', type=int, default=500, help='Frequency of printing training progress')

    def parse(self):
        self.opt = self.parser.parse_args()
        self.opt.is_train = True
        args = vars(self.opt)
        return self.opt
