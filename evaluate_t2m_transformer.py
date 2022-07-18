import os

from os.path import join as pjoin

import utils.paramUtil as paramUtil
from options.evaluate_options import TestT2MOptions
from utils.plot_script import *

from networks.transformer import TransformerV1, TransformerV2
from networks.quantizer import *
from networks.modules import *
from networks.trainers import TransformerT2MTrainer
from data.dataset import Motion2TextEvalDataset
from scripts.motion_process import *
from torch.utils.data import DataLoader
from utils.word_vectorizer import WordVectorizerV2


def plot_t2m(data, captions, save_dir):
    data = data * std + mean
    for i in range(len(data)):
        joint_data = data[i]
        caption = captions[i]
        joint = recover_from_ric(torch.from_numpy(joint_data).float(), opt.joints_num).numpy()
        # joint = motion_temporal_filter(joint)
        save_path = '%s_%02d.mp4' % (save_dir, i)
        np.save('%s_%02d.npy'%(save_dir, i), joint)
        plot_3d_motion(save_path, kinematic_chain, joint, title=caption, fps=fps, radius=radius)


def build_models(opt):
    vq_decoder = VQDecoderV3(opt.dim_vq_latent, dec_channels, opt.n_resblk, opt.n_down)
    quantizer = None
    if opt.q_mode == 'ema':
        quantizer = EMAVectorQuantizer(opt.codebook_size, opt.dim_vq_latent, opt.lambda_beta)
    elif opt.q_mode == 'cmt':
        quantizer = Quantizer(opt.codebook_size, opt.dim_vq_latent, opt.lambda_beta)

    checkpoint = torch.load(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.tokenizer_name, 'model', 'finest.tar'),
                            map_location=opt.device)
    vq_decoder.load_state_dict(checkpoint['vq_decoder'])
    quantizer.load_state_dict(checkpoint['quantizer'])

    if opt.t2m_v2:
        t2m_transformer = TransformerV2(n_txt_vocab, opt.txt_pad_idxt, n_mot_vocab, opt.mot_pad_idx, d_src_word_vec=512,
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
                                        n_dec_layers=opt.n_dec_layers, n_head=opt.n_head, d_k=opt.d_k, d_v=opt.d_v,
                                        dropout=0.1,
                                        n_src_position=50, n_trg_position=100,
                                        trg_emb_prj_weight_sharing=opt.proj_share_weight)
    checkpoint = torch.load(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name, 'model', '%s.tar'%(opt.which_epoch)),
                            map_location=opt.device)
    t2m_transformer.load_state_dict(checkpoint['t2m_transformer'])
    print('Loading t2m_transformer model: Epoch %03d Total_Iter %03d' % (checkpoint['ep'], checkpoint['total_it']))

    return vq_decoder, quantizer, t2m_transformer



if __name__ == '__main__':
    parser = TestT2MOptions()
    opt = parser.parse()

    opt.device = torch.device("cpu" if opt.gpu_id==-1 else "cuda:" + str(opt.gpu_id))
    torch.autograd.set_detect_anomaly(True)
    if opt.gpu_id != -1:
        torch.cuda.set_device(opt.gpu_id)

    opt.result_dir = pjoin(opt.result_path, opt.dataset_name, opt.name, opt.ext)
    opt.joint_dir = pjoin(opt.result_dir, 'joints')
    opt.animation_dir = pjoin(opt.result_dir, 'animations')

    os.makedirs(opt.joint_dir, exist_ok=True)
    os.makedirs(opt.animation_dir, exist_ok=True)

    if opt.dataset_name == 't2m':
        opt.data_root = '../text2motion/dataset/pose_data_raw/'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.m_token_dir = pjoin(opt.data_root, 'VQVAEV3_CB1024_CMT_H1024_NRES3')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 22
        opt.max_motion_token = 55
        opt.max_motion_frame = 196
        dim_pose = 263
        radius = 4
        fps = 20
        kinematic_chain = paramUtil.t2m_kinematic_chain
    elif opt.dataset_name == 'kit':
        opt.data_root = './dataset/kit_mocap_dataset'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.m_token_dir = pjoin(opt.data_root, 'VQVAEV3_CB1024_CMT_H1024_NRES3')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 21
        radius = 240 * 8
        fps = 12.5
        dim_pose = 251
        opt.max_motion_token = 55
        opt.max_motion_frame = 196
        kinematic_chain = paramUtil.kit_kinematic_chain
    else:
        raise KeyError('Dataset Does Not Exist')

    mean = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.tokenizer_name, 'meta', 'mean.npy'))
    std = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.tokenizer_name, 'meta', 'std.npy'))

    n_mot_vocab = opt.codebook_size + 3
    opt.mot_start_idx = opt.codebook_size
    opt.mot_end_idx = opt.codebook_size + 1
    opt.mot_pad_idx = opt.codebook_size + 2

    enc_channels = [1024, opt.dim_vq_latent]
    dec_channels = [opt.dim_vq_latent, 1024, dim_pose]

    w_vectorizer = WordVectorizerV2('../text2motion/glove', 'our_vab')
    n_txt_vocab = len(w_vectorizer) + 1
    _, _, opt.txt_start_idx = w_vectorizer['sos/OTHER']
    _, _, opt.txt_end_idx = w_vectorizer['eos/OTHER']
    opt.txt_pad_idx = len(w_vectorizer)

    vq_decoder, quantizer, t2m_transformer = build_models(opt)

    split_file = pjoin(opt.data_root, opt.split_file)


    dataset = Motion2TextEvalDataset(opt, mean, std, split_file, w_vectorizer)
    data_loader = DataLoader(dataset, batch_size=opt.batch_size,num_workers=1, shuffle=True, pin_memory=True)

    vq_decoder.to(opt.device)
    quantizer.to(opt.device)
    t2m_transformer.to(opt.device)

    vq_decoder.eval()
    quantizer.eval()
    t2m_transformer.eval()

    opt.repeat_times = opt.repeat_times if opt.sample else 1

    '''Generating Results'''
    print('Generating Results')
    result_dict = {}
    with torch.no_grad():
        for i, batch_data in enumerate(data_loader):
            print('%02d_%03d'%(i, opt.num_results))
            word_emb, pos_ohot, captions, sent_lens, motions, m_tokens, m_lens, _ = batch_data

            # word_emb, word_ids, caption, cap_lens, m_tokens, len_tokens = batch_data
            word_emb = word_emb.detach().to(opt.device).float()
            m_tokens = m_tokens.detach().to(opt.device).long()
            # word_ids = word_ids.detach().to(opt.device).long()
            # gt_tokens = motions[:, :m_lens[0]]

            print(captions[0])
            # print('Ground Truth Tokens')
            # print(gt_tokens[0])

            # rec_vq_latent = quantizer.get_codebook_entry(gt_tokens)
            # rec_motion = vq_decoder(rec_vq_latent)

            name = 'L%03dC%03d' % (m_lens[0], i)
            item_dict = {
                'caption': captions,
                'length': m_lens[0],
                'gt_motion': motions[:, :m_lens[0]].cpu().numpy()
            }
            for t in range(opt.repeat_times):
                # if opt.t2m_v2:
                #     pred_tokens = t2m_transformer.sample(word_ids, trg_sos=opt.mot_start_idx, trg_eos=opt.mot_end_idx,
                #                                          max_steps=80, sample=opt.sample, top_k=opt.top_k)
                # else:
                pred_tokens = t2m_transformer.sample(word_emb, sent_lens, trg_sos=opt.mot_start_idx,
                                                     trg_eos=opt.mot_end_idx, max_steps=80, sample=opt.sample,
                                                     top_k=opt.top_k)
                pred_tokens = pred_tokens[:, 1:]
                print('Sampled Tokens %02d'%t)
                print(pred_tokens[0])
                if len(pred_tokens[0]) == 0:
                    continue
                vq_latent = quantizer.get_codebook_entry(pred_tokens)
                gen_motion = vq_decoder(vq_latent)

                sub_dict = {}
                sub_dict['motion'] = gen_motion.cpu().numpy()
                sub_dict['length'] = len(gen_motion[0])
                item_dict['result_%02d'%t] = sub_dict

            result_dict[name] = item_dict
            if i > opt.num_results:
                break

    print('Animating Results')
    '''Animating Results'''
    for i, (key, item) in enumerate(result_dict.items()):
        print('%02d_%03d' % (i, opt.num_results))
        captions = item['caption']
        gt_motions = item['gt_motion']
        joint_save_path = pjoin(opt.joint_dir, key)
        animation_save_path = pjoin(opt.animation_dir, key)

        os.makedirs(joint_save_path, exist_ok=True)
        os.makedirs(animation_save_path, exist_ok=True)

        # np.save(pjoin(joint_save_path, 'gt_motions.npy'), gt_motions)
        plot_t2m(gt_motions, captions, pjoin(animation_save_path, 'gt_motion'))
        for t in range(opt.repeat_times):
            sub_dict = item['result_%02d' % t]
            motion = sub_dict['motion']
            # np.save(pjoin(joint_save_path, 'gen_motion_%02d_L%03d.npy' % (t, motion.shape[1])), motion)
            plot_t2m(motion, captions, pjoin(animation_save_path, 'gen_motion_%02d_L%03d' % (t, motion.shape[1])))