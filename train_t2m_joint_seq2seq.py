import os

from os.path import join as pjoin

import utils.paramUtil as paramUtil
from options.train_options import TrainT2MOptions
from utils.plot_script import *

from networks.transformer import TransformerV1, TransformerV2
from networks.quantizer import *
from networks.modules import *
from networks.trainers import Seq2SeqT2MJointTrainer
from data.dataset import TextMotionTokenDataset, collate_fn
from scripts.motion_process import *
from torch.utils.data import DataLoader
from utils.word_vectorizer import WordVectorizerV2


def build_models(opt):
    t2m_model = Seq2SeqText2MotModel(300, opt.n_mot_vocab, opt.dim_txt_hid, opt.dim_mot_hid,
                                     opt.n_mot_layers, opt.device, opt.early_or_late)

    if opt.dataset_name == "t2m":
        m2t_transformer = TransformerV2(opt.n_mot_vocab, opt.mot_pad_idx, opt.n_txt_vocab, opt.txt_pad_idx, d_src_word_vec=512,
                                        d_trg_word_vec=512,
                                        d_model=opt.d_model, d_inner=opt.d_inner_hid, n_enc_layers=4,
                                        n_dec_layers=4, n_head=opt.n_head, d_k=opt.d_k, d_v=opt.d_v,
                                        dropout=0.1,
                                        n_src_position=100, n_trg_position=50,
                                        trg_emb_prj_weight_sharing=opt.proj_share_weight
                                        )
        checkpoint = torch.load(pjoin(opt.checkpoints_dir, opt.dataset_name,'M2T_EL4_DL4_NH8_PS' , 'model', 'finest.tar'),
                                map_location=opt.device)
        m2t_transformer.load_state_dict(checkpoint['m2t_transformer'])
    elif opt.dataset_name == "kit":
        m2t_transformer = TransformerV2(opt.n_mot_vocab, opt.mot_pad_idx, opt.n_txt_vocab, opt.txt_pad_idx,
                                        d_src_word_vec=512,
                                        d_trg_word_vec=512,
                                        d_model=opt.d_model, d_inner=opt.d_inner_hid, n_enc_layers=3,
                                        n_dec_layers=3, n_head=opt.n_head, d_k=opt.d_k, d_v=opt.d_v,
                                        dropout=0.1,
                                        n_src_position=100, n_trg_position=50,
                                        trg_emb_prj_weight_sharing=opt.proj_share_weight
                                        )
        checkpoint = torch.load(
            pjoin(opt.checkpoints_dir, opt.dataset_name, 'M2T_EL3_DL3_NH8_PS', 'model', 'finest.tar'),
            map_location=opt.device)
        m2t_transformer.load_state_dict(checkpoint['m2t_transformer'])
    print('Loading m2t_transformer model: Epoch %03d Total_Iter %03d' % (checkpoint['ep'], checkpoint['total_it']))

    return t2m_model, m2t_transformer


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
        opt.data_root = './dataset/HumanML3D/'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.joints_num = 22
        opt.max_motion_len = 55
        dim_pose = 263
        radius = 4
        fps = 20
        kinematic_chain = paramUtil.t2m_kinematic_chain
    elif opt.dataset_name == 'kit':
        opt.data_root = './dataset/KIT-ML/'
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

    w_vectorizer = WordVectorizerV2('./glove', 'our_vab')

    opt.n_mot_vocab = opt.codebook_size + 3
    opt.mot_start_idx = opt.codebook_size
    opt.mot_end_idx = opt.codebook_size + 1
    opt.mot_pad_idx = opt.codebook_size + 2

    opt.n_txt_vocab = len(w_vectorizer) + 1
    _, _, opt.txt_start_idx = w_vectorizer['sos/OTHER']
    _, _, opt.txt_end_idx = w_vectorizer['eos/OTHER']
    opt.txt_pad_idx = len(w_vectorizer)


    enc_channels = [1024, opt.dim_vq_latent]
    dec_channels = [opt.dim_vq_latent, 1024, dim_pose]

    t2m_model, m2t_transformer = build_models(opt)


    all_params = 0
    pc_transformer = sum(param.numel() for param in t2m_model.parameters())
    print(t2m_model)
    print("Total parameters of t2m_transformer net: {}".format(pc_transformer))
    # print(m2t_transformer)
    # print("Total parameters of m2t_transformer net: {}".format(sum(param.numel() for param in m2t_transformer.parameters())))
    all_params += pc_transformer

    print('Total parameters of all models: {}'.format(all_params))

    trainer = Seq2SeqT2MJointTrainer(opt, t2m_model, m2t_transformer)

    train_dataset = TextMotionTokenDataset(opt, train_split_file, w_vectorizer)
    val_dataset = TextMotionTokenDataset(opt, val_split_file, w_vectorizer)

    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, drop_last=False, num_workers=4,
                              shuffle=True, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, drop_last=False, num_workers=4,
                            shuffle=True, pin_memory=True, collate_fn=collate_fn)

    trainer.train(train_loader, val_loader, w_vectorizer)