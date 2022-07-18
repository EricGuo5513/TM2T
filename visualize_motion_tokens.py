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
from utils.utils import *


def plot_t2m(data, count):
    data = data * std + mean
    for i in range(len(data)):
        joint_data = data[i]
        joint = recover_from_ric(torch.from_numpy(joint_data).float(), opt.joints_num).numpy()
        # joint = motion_temporal_filter(joint)
        # save_path = '%s_%02d.mp4' % (save_dir, i)
        np.save(pjoin(opt.joint_dir, "%d.npy"%count), joint)
        plot_3d_motion(pjoin(opt.animation_dir, "%d.mp4"%count),
                       kinematic_chain, joint, title="None", fps=fps, radius=radius)


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
    return vq_decoder, quantizer



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
        opt.joints_num = 22
        opt.max_motion_token = 55
        opt.max_motion_frame = 196
        dim_pose = 263
        radius = 4
        fps = 20
        kinematic_chain = paramUtil.t2m_kinematic_chain
    elif opt.dataset_name == 'kit':
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

    enc_channels = [1024, opt.dim_vq_latent]
    dec_channels = [opt.dim_vq_latent, 1024, dim_pose]


    vq_decoder, quantizer = build_models(opt)

    vq_decoder.to(opt.device)
    quantizer.to(opt.device)

    vq_decoder.eval()
    quantizer.eval()

    with torch.no_grad():
        for i in tqdm(range(1024)):
            m_token = torch.LongTensor(1, 1).fill_(i).to(opt.device)
            vq_latent = quantizer.get_codebook_entry(m_token)
            gen_motion = vq_decoder(vq_latent)

            plot_t2m(gen_motion.cpu().numpy(), i)

