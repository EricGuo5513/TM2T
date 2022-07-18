import os

from os.path import join as pjoin

import utils.paramUtil as paramUtil
from options.train_options import TrainT2MOptions
from utils.plot_script import *

from networks.transformer import TransformerV1, TransformerV2
from networks.quantizer import *
from networks.modules import *
from networks.trainers import TransformerT2MTrainer
from data.dataset import TextMotionTokenDataset
from scripts.motion_process import *
from torch.utils.data import DataLoader
from utils.word_vectorizer import WordVectorizerV2


if __name__ == '__main__':
    parser = TrainT2MOptions()
    opt = parser.parse()

    opt.device = torch.device("cpu" if opt.gpu_id==-1 else "cuda:" + str(opt.gpu_id))
    torch.autograd.set_detect_anomaly(True)
    if opt.gpu_id != -1:
        torch.cuda.set_device(opt.gpu_id)

    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.meta_dir = pjoin(opt.save_root, 'meta')
    opt.eval_dir = pjoin(opt.save_root, 'animation')
    opt.log_dir = pjoin('./log', opt.dataset_name, opt.name)

    os.makedirs(opt.model_dir, exist_ok=True)
    os.makedirs(opt.meta_dir, exist_ok=True)
    os.makedirs(opt.eval_dir, exist_ok=True)
    os.makedirs(opt.log_dir, exist_ok=True)

    if opt.dataset_name == 't2m':
        opt.data_root = '../text2motion/dataset/pose_data_raw/'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.joints_num = 22
        opt.max_motion_len = 55
        dim_pose = 263
        radius = 4
        fps = 20
        kinematic_chain = paramUtil.t2m_kinematic_chain
    elif opt.dataset_name == 'kit':
        opt.data_root = '../text2motion/dataset/kit_mocap_dataset'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.joints_num = 21
        radius = 240 * 8
        fps = 12.5
        dim_pose = 251
        opt.max_motion_len = 55
        kinematic_chain = paramUtil.kit_kinematic_chain
    else:
        raise KeyError('Dataset Does Not Exist')

    if opt.text_aug:
        opt.text_dir = pjoin(opt.data_root, '%s_AUG_texts'%(opt.tokenizer_name))

    else:
        opt.text_dir = pjoin(opt.data_root, 'texts')


    mean = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.tokenizer_name, 'meta', 'mean.npy'))
    std = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.tokenizer_name, 'meta', 'std.npy'))

    train_split_file = pjoin(opt.data_root, 'train.txt')
    val_split_file = pjoin(opt.data_root, 'val.txt')

    w_vectorizer = WordVectorizerV2('../text2motion/glove', 'our_vab')

    n_mot_vocab = opt.codebook_size + 3
    opt.mot_start_idx = opt.codebook_size
    opt.mot_end_idx = opt.codebook_size + 1
    opt.mot_pad_idx = opt.codebook_size + 2

    n_txt_vocab = len(w_vectorizer) + 1
    _, _, opt.txt_start_idx = w_vectorizer['sos/OTHER']
    _, _, opt.txt_end_idx = w_vectorizer['eos/OTHER']
    opt.txt_pad_idx = len(w_vectorizer)


    enc_channels = [1024, opt.dim_vq_latent]
    dec_channels = [opt.dim_vq_latent, 1024, dim_pose]

    if opt.t2m_v2:
        t2m_transformer = TransformerV2(n_txt_vocab, opt.txt_pad_idx, n_mot_vocab, opt.mot_pad_idx, d_src_word_vec=512,
                                        d_trg_word_vec=512,
                                        d_model=opt.d_model, d_inner=opt.d_inner_hid, n_enc_layers=opt.n_enc_layers,
                                        n_dec_layers=opt.n_dec_layers, n_head=opt.n_head, d_k=opt.d_k, d_v=opt.d_v,
                                        dropout=0.1,
                                        n_src_position=50, n_trg_position=100,
                                        trg_emb_prj_weight_sharing=opt.proj_share_weight
                                        )
    else:
        t2m_transformer = TransformerV1(n_mot_vocab, opt.mot_pad_idx, d_src_word_vec=300, d_trg_word_vec=512,
                                        d_model=opt.d_model, d_inner=opt.d_inner_hid, n_enc_layers=opt.n_enc_layers,
                                        n_dec_layers=opt.n_dec_layers, n_head=opt.n_head, d_k=opt.d_k, d_v=opt.d_v, dropout=0.1,
                                        n_src_position=50, n_trg_position=100, trg_emb_prj_weight_sharing=opt.proj_share_weight)


    all_params = 0
    pc_transformer = sum(param.numel() for param in t2m_transformer.parameters())
    print(t2m_transformer)
    print("Total parameters of t2m_transformer net: {}".format(pc_transformer))
    all_params += pc_transformer

    print('Total parameters of all models: {}'.format(all_params))

    trainer = TransformerT2MTrainer(opt, t2m_transformer)

    train_dataset = TextMotionTokenDataset(opt, train_split_file, w_vectorizer)
    val_dataset = TextMotionTokenDataset(opt, val_split_file, w_vectorizer)

    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, drop_last=True, num_workers=4,
                              shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, drop_last=True, num_workers=4,
                            shuffle=True, pin_memory=True)

    trainer.train(train_loader, val_loader, None)