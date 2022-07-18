import os

from os.path import join as pjoin

import utils.paramUtil as paramUtil
from options.evaluate_options import TestT2MOptions
from utils.plot_script import *

from networks.transformer import TransformerV1, TransformerV2
from networks.quantizer import *
from networks.modules import *
from networks.trainers import TransformerT2MTrainer
from data.dataset import RawTextDataset
from scripts.motion_process import *
from torch.utils.data import DataLoader
from utils.word_vectorizer import WordVectorizerV2
from utils.utils import *


def plot_t2m(data, captions, save_dir):
    data = data * std + mean
    for i in range(len(data)):
        joint_data = data[i]
        caption = captions[i]
        joint = recover_from_ric(torch.from_numpy(joint_data).float(), opt.joints_num).numpy()
        joint = motion_temporal_filter(joint)
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

    t2m_model = Seq2SeqText2MotModel(300, n_mot_vocab, opt.dim_txt_hid, opt.dim_mot_hid,
                                     opt.n_mot_layers, opt.device, opt.early_or_late)
    checkpoint = torch.load(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name, 'model', 'finest.tar'),
                            map_location=opt.device)
    t2m_model.load_state_dict(checkpoint['t2m_model'])
    print('Loading t2m_model model: Epoch %03d Total_Iter %03d' % (checkpoint['ep'], checkpoint['total_it']))

    return vq_decoder, quantizer, t2m_model



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
        # opt.data_root = '../text2motion/dataset/pose_data_raw/'
        # opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        # opt.m_token_dir = pjoin(opt.data_root, 'VQVAEV3_CB1024_CMT_H1024_NRES3')
        # opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 22
        opt.max_motion_token = 55
        opt.max_motion_frame = 196
        dim_pose = 263
        radius = 4
        fps = 20
        kinematic_chain = paramUtil.t2m_kinematic_chain
    elif opt.dataset_name == 'kit':
        # opt.data_root = './dataset/kit_mocap_dataset'
        # opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        # opt.m_token_dir = pjoin(opt.data_root, 'VQVAEV3_CB1024_CMT_H1024_NRES3')
        # opt.text_dir = pjoin(opt.data_root, 'texts')
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

    vq_decoder, quantizer, t2m_model = build_models(opt)

    # split_file = pjoin(opt.data_root, opt.split_file)


    dataset = RawTextDataset(opt, mean, std, opt.text_file, w_vectorizer)
    data_loader = DataLoader(dataset, batch_size=opt.batch_size,num_workers=1, pin_memory=True)

    vq_decoder.to(opt.device)
    quantizer.to(opt.device)
    t2m_model.to(opt.device)

    vq_decoder.eval()
    quantizer.eval()
    t2m_model.eval()

    opt.repeat_times = opt.repeat_times if opt.sample else 1

    '''Generating Results'''
    print('Generating Results')
    result_dict = {}
    with torch.no_grad():
        for i, batch_data in enumerate(data_loader):
            print('%02d_%03d'%(i, opt.num_results))
            word_emb, pos_ohot, captions, cap_lens = batch_data

            # word_emb, word_ids, caption, cap_lens, m_tokens, len_tokens = batch_data
            word_emb = word_emb.detach().to(opt.device).float()
            # m_tokens = m_tokens.detach().to(opt.device).long()
            # word_ids = word_ids.detach().to(opt.device).long()
            # gt_tokens = motions[:, :m_lens[0]]

            print(captions[0])
            # print('Ground Truth Tokens')
            # print(gt_tokens[0])

            # rec_vq_latent = quantizer.get_codebook_entry(gt_tokens)
            # rec_motion = vq_decoder(rec_vq_latent)

            name = 'C%03d' % (i)
            item_dict = {
                'caption': captions,
                # 'length': m_lens[0],
                # 'gt_motion': motions[:, :m_lens[0]].cpu().numpy()
            }
            for t in range(opt.repeat_times):
                pred_tokens, len_map = t2m_model.sample_batch(word_emb, cap_lens, trg_sos=opt.mot_start_idx,
                                                              trg_eos=opt.mot_end_idx, max_steps=49, top_k=opt.top_k)
                # print(pred_tokens)
                pred_tokens = pred_tokens[:, 1:len_map[0]+1]
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
        print('%02d_%03d'%(i, opt.num_results))
        captions = item['caption']
        # gt_motions = item['gt_motion']
        joint_save_path = pjoin(opt.joint_dir, key)
        animation_save_path = pjoin(opt.animation_dir, key)

        os.makedirs(joint_save_path, exist_ok=True)
        os.makedirs(animation_save_path, exist_ok=True)

        # np.save(pjoin(joint_save_path, 'gt_motions.npy'), gt_motions)
        # plot_t2m(gt_motions, captions, pjoin(animation_save_path, 'gt_motion'))
        for t in range(opt.repeat_times):
            sub_dict = item['result_%02d'%t]
            motion = sub_dict['motion']
            # np.save(pjoin(joint_save_path, 'gen_motion_%02d_L%03d.npy' % (t, motion.shape[1])), motion)
            plot_t2m(motion, captions, pjoin(animation_save_path, 'gen_motion_%02d_L%03d' % (t, motion.shape[1])))