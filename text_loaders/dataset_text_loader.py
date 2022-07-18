from data.dataset import Motion2TextEvalDataset, collate_fn
from utils.word_vectorizer import WordVectorizerV2
import numpy as np
from os.path import join as pjoin
from torch.utils.data import DataLoader
from utils.get_opt import get_opt


def get_dataset_text_loader(opt_path, batch_size, device):
    opt = get_opt(opt_path, device)

    # Configurations of T2M dataset and KIT dataset is almost the same
    if opt.dataset_name == 't2m' or opt.dataset_name == 'kit':
        print('Loading dataset %s ...' % opt.dataset_name)

        mean = np.load(pjoin(opt.meta_dir, 'mean.npy'))
        std = np.load(pjoin(opt.meta_dir, 'std.npy'))

        w_vectorizer = WordVectorizerV2('./glove', 'our_vab')
        split_file = pjoin(opt.data_root, 'test.txt')
        opt.m_token_dir = pjoin(opt.data_root, 'VQVAEV3_CB1024_CMT_H1024_NRES3')
        opt.n_mot_vocab = 1024 + 3
        opt.mot_start_idx = 1024
        opt.mot_end_idx = 1024 + 1
        opt.mot_pad_idx = 1024 + 2

        dataset = Motion2TextEvalDataset(opt, mean, std, split_file, w_vectorizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, drop_last=True,
                                collate_fn=collate_fn, shuffle=True)
    else:
        raise KeyError('Dataset not Recognized !!')

    print('Ground Truth Dataset Loading Completed!!!')
    return dataloader, dataset