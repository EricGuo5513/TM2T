import torch
from networks.modules import *
from networks.transformer import TransformerV1, TransformerV2
from networks.quantizer import *

# from networks.trainers import CompTrainerV6
from torch.utils.data import Dataset, DataLoader
from os.path import join as pjoin
from tqdm import tqdm
from data.dataset import collate_fn


def build_models(opt):
    vq_decoder = VQDecoderV3(opt.dim_vq_latent, opt.dec_channels, opt.n_resblk, opt.n_down)
    quantizer = None
    if opt.q_mode == 'ema':
        quantizer = EMAVectorQuantizer(opt.codebook_size, opt.dim_vq_latent, opt.lambda_beta)
    elif opt.q_mode == 'cmt':
        quantizer = Quantizer(opt.codebook_size, opt.dim_vq_latent, opt.lambda_beta)

    checkpoint = torch.load(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.tokenizer_name, 'model', 'finest.tar'),
                            map_location=opt.device)
    vq_decoder.load_state_dict(checkpoint['vq_decoder'])
    quantizer.load_state_dict(checkpoint['quantizer'])

    t2m_model = Seq2SeqText2MotModel(300, opt.n_mot_vocab, opt.dim_txt_hid, opt.dim_mot_hid,
                                     opt.n_mot_layers, opt.device, opt.early_or_late)
    checkpoint = torch.load(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name, 'model', 'finest.tar'),
                            map_location=opt.device)
    t2m_model.load_state_dict(checkpoint['t2m_model'])
    print('Loading t2m_model model: Epoch %03d Total_Iter %03d' % (checkpoint['ep'], checkpoint['total_it']))

    return vq_decoder, quantizer, t2m_model


class T2MSeq2SeqGeneratedDataset(Dataset):

    def __init__(self, opt, dataset, w_vectorizer, mm_num_samples, mm_num_repeats):
        assert mm_num_samples < len(dataset)
        print(opt.model_dir)

        if opt.dataset_name == 't2m':
            opt.max_motion_len = 55
            dim_pose = 263
        elif opt.dataset_name == 'kit':
            dim_pose = 251
            opt.max_motion_len = 55
        else:
            raise KeyError('Dataset Does Not Exist')
        batch_size = 128

        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, collate_fn=collate_fn, shuffle=True)
        opt.n_mot_vocab = opt.codebook_size + 3
        opt.mot_start_idx = opt.codebook_size
        opt.mot_end_idx = opt.codebook_size + 1
        opt.mot_pad_idx = opt.codebook_size + 2

        opt.enc_channels = [1024, opt.dim_vq_latent]
        opt.dec_channels = [opt.dim_vq_latent, 1024, dim_pose]

        opt.n_txt_vocab = len(w_vectorizer) + 1
        _, _, opt.txt_start_idx = w_vectorizer['sos/OTHER']
        _, _, opt.txt_end_idx = w_vectorizer['eos/OTHER']
        opt.txt_pad_idx = len(w_vectorizer)

        vq_decoder, quantizer, t2m_model = build_models(opt)

        # movement_enc, movement_dec = loadDecompModel(opt)

        generated_motion = []
        mm_generated_motions = []
        mm_idxs = np.random.choice(len(dataset) // batch_size + 1, mm_num_samples // batch_size + 1, replace=False)
        mm_idxs = np.sort(mm_idxs)
        # print(mm_idxs)

        vq_decoder.to(opt.device)
        quantizer.to(opt.device)
        t2m_model.to(opt.device)

        vq_decoder.eval()
        quantizer.eval()
        t2m_model.eval()

        # movement_enc.to(opt.device)
        # movement_dec.to(opt.device)
        #
        # movement_enc.eval()
        # movement_dec.eval()

        with torch.no_grad():
            for i, data in tqdm(enumerate(dataloader)):
                word_emb, pos_ohot, caption, cap_lens, motions, m_lens, tokens = data
                # print(tokens)
                word_emb = word_emb.detach().to(opt.device).float()

                mm_num_now = len(mm_generated_motions)
                is_mm = True if ((mm_num_now < mm_num_samples) and (i == mm_idxs[mm_num_now])) else False
                # if is_mm:
                #     print(mm_num_now, i, mm_idxs[mm_num_now])
                repeat_times = mm_num_repeats if is_mm else 1
                mm_batch_list = []
                # print(m_lens[0].item(), cap_lens[0].item())
                for t in range(repeat_times):
                    pred_tokens, len_map = t2m_model.sample_batch(word_emb, cap_lens, trg_sos=opt.mot_start_idx,
                                                                        trg_eos=opt.mot_end_idx, max_steps=49, top_k=100)

                    # print(pred_tokens.shape)
                    # print(len_map)
                    c = 0
                    for i in range(len(pred_tokens)):
                        m_tokens = pred_tokens[i:i+1]
                        m_len = len_map[i]
                        if m_len == 0:
                            continue
                        m_tokens = m_tokens[:, 1:m_len+1]
                        # print(m_tokens)
                        vq_latent = quantizer.get_codebook_entry(m_tokens)
                        pred_motions = vq_decoder(vq_latent)

                        token = tokens[i].split('_')
                        # print(pred_motions)
                        if t == 0:
                            # print(m_lens)
                            # print(text_data)
                            # pred_motions = movement_dec(movement_enc(pred_motions[..., :-4]))

                            sub_dict = {'motion': pred_motions[0].cpu().numpy(),
                                        'length': pred_motions.shape[1],
                                        'cap_len': cap_lens[i].item(),
                                        'caption': caption[i],
                                        'tokens': token}
                            generated_motion.append(sub_dict)

                        if is_mm:
                            if t == 0 or i >= len(mm_batch_list):
                                mm_batch_list.append({
                                    'caption':caption[i],
                                    'tokens': token,
                                    'cap_len': cap_lens[i].item(),
                                    'mm_motions': [{
                                        'motion': pred_motions[0].cpu().numpy(),
                                        'length': pred_motions.shape[1]
                                    }]
                                })
                            else:
                                mm_batch_list[i]['mm_motions'].append({
                                    'motion': pred_motions[0].cpu().numpy(),
                                    'length': pred_motions.shape[1]
                                })
                if is_mm:
                    mm_generated_motions.extend(mm_batch_list)

                    # if len(mm_generated_motions) < mm_num_samples:
                    #     print(len(mm_generated_motions), mm_idxs[len(mm_generated_motions)])
        self.generated_motion = generated_motion
        self.mm_generated_motion = mm_generated_motions
        # print(len(generated_motion))
        # print(len(mm_generated_motions))
        self.opt = opt
        self.w_vectorizer = w_vectorizer


    def __len__(self):
        return len(self.generated_motion)


    def __getitem__(self, item):
        data = self.generated_motion[item]
        motion, m_length, caption, tokens = data['motion'], data['length'], data['caption'], data['tokens']
        sent_len = data['cap_len']
        # tokens = text_data['tokens']
        # if len(tokens) < self.opt.max_text_len:
        #     # pad with "unk"
        #     tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
        #     sent_len = len(tokens)
        #     tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
        # else:
        #     # crop
        #     tokens = tokens[:self.opt.max_text_len]
        #     tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
        #     sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh, _ = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        # print(tokens)
        # print(caption)
        # print(m_length)
        # print(self.opt.max_motion_length)
        if m_length < self.opt.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.opt.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)
        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens)