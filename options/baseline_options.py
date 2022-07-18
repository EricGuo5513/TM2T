import argparse
import torch
import os

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--name', type=str, default="test", help='Name of this trial')
        self.parser.add_argument("--gpu_id", type=int, default=-1, help='GPU id')

        self.parser.add_argument('--dataset_name', type=str, default='t2m', help='Dataset Name')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')

        self.parser.add_argument("--unit_length", type=int, default=4, help="Length of motion")
        self.parser.add_argument("--max_text_len", type=int, default=20, help="Length of motion")

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


class BaselineSeq2SeqOptions(BaseOptions):
    def __init__(self):
        super(BaselineSeq2SeqOptions, self).__init__()
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        self.parser.add_argument('--dim_mot_hid', type=int, default=512, help='Dimension of hidden unit in GRU')
        self.parser.add_argument('--dim_txt_hid', type=int, default=512, help='Dimension of hidden unit in GRU')

        self.parser.add_argument('--n_txt_layers', type=int, default=2, help='Dimension of hidden unit in GRU')

        self.parser.add_argument('--is_continue', action="store_true", help='Training iterations')
        self.parser.add_argument('--use_att', action="store_true", help='Training iterations')

        self.parser.add_argument('--tf_ratio', type=float, default=0.4, help='Training iterations')

        self.parser.add_argument('--label_smoothing', action='store_true')
        self.parser.add_argument('--max_epoch', type=int, default=300, help='Training iterations')
        self.parser.add_argument('--batch_size', type=int, default=64, help='Batch size')

        self.parser.add_argument('--lr', type=float, default=2e-4, help='Layers of GRU')
        self.parser.add_argument('--log_every', type=int, default=50, help='Frequency of printing training progress')
        self.parser.add_argument('--save_every_e', type=int, default=10, help='Frequency of printing training progress')
        self.parser.add_argument('--eval_every_e', type=int, default=5, help='Frequency of printing training progress')
        self.parser.add_argument('--save_latest', type=int, default=500, help='Frequency of printing training progress')

        self.is_train = True

class BaselineSeq2SeqGANOptions(BaseOptions):
    def __init__(self):
        super(BaselineSeq2SeqGANOptions, self).__init__()
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        self.parser.add_argument('--dim_mot_hid', type=int, default=512, help='Dimension of hidden unit in GRU')
        self.parser.add_argument('--dim_txt_hid', type=int, default=512, help='Dimension of hidden unit in GRU')

        self.parser.add_argument('--n_txt_layers', type=int, default=2, help='Dimension of hidden unit in GRU')

        self.parser.add_argument('--is_continue', action="store_true", help='Training iterations')
        self.parser.add_argument('--use_att', action="store_true", help='Training iterations')

        self.parser.add_argument('--start_gan', action="store_true", help='Training iterations')

        self.parser.add_argument('--tf_ratio', type=float, default=0.4, help='Training iterations')

        self.parser.add_argument('--label_smoothing', action='store_true')
        self.parser.add_argument('--max_epoch', type=int, default=300, help='Training iterations')
        self.parser.add_argument('--batch_size', type=int, default=64, help='Batch size')

        self.parser.add_argument('--lr', type=float, default=2e-4, help='Layers of GRU')
        self.parser.add_argument('--log_every', type=int, default=50, help='Frequency of printing training progress')
        self.parser.add_argument('--save_every_e', type=int, default=10, help='Frequency of printing training progress')
        self.parser.add_argument('--eval_every_e', type=int, default=5, help='Frequency of printing training progress')
        self.parser.add_argument('--save_latest', type=int, default=500, help='Frequency of printing training progress')

        self.is_train = True
