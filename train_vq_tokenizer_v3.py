import os

from os.path import join as pjoin

import utils.paramUtil as paramUtil
from options.train_options import TrainVQTokenizerOptions
from utils.plot_script import *

from networks.modules import *
from networks.quantizer import *
from networks.trainers import VQTokenizerTrainerV3
from data.dataset import MotionDataset
from scripts.motion_process import *
from torch.utils.data import DataLoader


def plot_t2m(data, save_dir):
    data = train_dataset.inv_transform(data)
    for i in range(len(data)):
        joint_data = data[i]
        joint = recover_from_ric(torch.from_numpy(joint_data).float(), opt.joints_num).numpy()
        save_path = pjoin(save_dir, '%02d.mp4' % (i))
        plot_3d_motion(save_path, kinematic_chain, joint, title="None", fps=fps, radius=radius)


def load_models(opt):
    vq_encoder = VQEncoderV3(dim_pose - 4, enc_channels, opt.n_down)
    vq_decoder = VQDecoderV3(opt.dim_vq_latent, dec_channels, opt.n_resblk, opt.n_down)

    quantizer = Quantizer(opt.codebook_size, opt.dim_vq_latent, opt.lambda_beta)

    # if not opt.is_continue:
    #     checkpoint = torch.load(pjoin(opt.checkpoints_dir, opt.dataset_name,
    #                                   'VQVAEV3_CB1024_CMT_H1024_NRES3', 'model', 'finest.tar'),
    #                             map_location=opt.device)
    #     vq_encoder.load_state_dict(checkpoint['vq_encoder'])
    #     vq_decoder.load_state_dict(checkpoint['vq_decoder'])
    #     quantizer.load_state_dict(checkpoint['quantizer'])
    return vq_encoder, vq_decoder, quantizer


if __name__ == '__main__':
    parser = TrainVQTokenizerOptions()
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
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 22
        opt.max_motion_length = 196
        dim_pose = 263
        radius = 4
        fps = 20
        kinematic_chain = paramUtil.t2m_kinematic_chain
    elif opt.dataset_name == 'kit':
        opt.data_root = './dataset/KIT-ML/'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 21
        radius = 240 * 8
        fps = 12.5
        dim_pose = 251
        opt.max_motion_length = 196
        kinematic_chain = paramUtil.kit_kinematic_chain
    else:
        raise KeyError('Dataset Does Not Exist')

    mean = np.load(pjoin(opt.data_root, 'Mean.npy'))
    std = np.load(pjoin(opt.data_root, 'Std.npy'))

    train_split_file = pjoin(opt.data_root, 'train.txt')
    val_split_file = pjoin(opt.data_root, 'val.txt')

    enc_channels = [1024, opt.dim_vq_latent]
    dec_channels = [opt.dim_vq_latent, 1024, dim_pose]

    # vq_encoder = VQEncoderV2(dim_pose-4, opt.dim_vq_enc_hidden, opt.dim_vq_latent

    vq_encoder, vq_decoder, quantizer = load_models(opt)

    # for name, parameters in vq_decoder.named_parameters():
    #     print(name, ':', parameters.size())
        # parm[name] = parameters.detach().numpy()

    discriminator = VQDiscriminator(dim_pose, opt.dim_vq_dis_hidden, opt.n_layers_dis)

    all_params = 0
    pc_vq_enc = sum(param.numel() for param in vq_encoder.parameters())
    print(vq_encoder)
    print("Total parameters of encoder net: {}".format(pc_vq_enc))
    all_params += pc_vq_enc

    pc_quan = sum(param.numel() for param in quantizer.parameters())
    print(quantizer)
    print("Total parameters of codebook: {}".format(pc_quan))
    all_params += pc_quan

    pc_vq_dec = sum(param.numel() for param in vq_decoder.parameters())
    print(vq_decoder)
    print("Total parameters of decoder net: {}".format(pc_vq_dec))
    all_params += pc_vq_dec

    pc_vq_dis = sum(param.numel() for param in discriminator.parameters())
    print(discriminator)
    print("Total parameters of discriminator net: {}".format(pc_vq_dis))
    all_params += pc_vq_dis

    print('Total parameters of all models: {}'.format(all_params))

    trainer = VQTokenizerTrainerV3(opt, vq_encoder, quantizer, vq_decoder, discriminator)

    train_dataset = MotionDataset(opt, mean, std, train_split_file)
    val_dataset = MotionDataset(opt, mean, std, val_split_file)

    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, drop_last=True, num_workers=4,
                              shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, drop_last=True, num_workers=4,
                            shuffle=True, pin_memory=True)

    trainer.train(train_loader, val_loader, plot_t2m)