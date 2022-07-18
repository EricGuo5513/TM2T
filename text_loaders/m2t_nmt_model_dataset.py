import torch
from networks.modules import *
from networks.transformer import TransformerV2
from networks.translator import Translator

# from networks.trainers import CompTrainerV6
from torch.utils.data import Dataset, DataLoader
from os.path import join as pjoin
from tqdm import tqdm
import spacy

# from data.dataset import collate_fn


def build_models(opt):
    m2t_transformer = TransformerV2(opt.n_mot_vocab, opt.mot_pad_idx, opt.n_txt_vocab, opt.txt_pad_idx, d_src_word_vec=512,
                                    d_trg_word_vec=512,
                                    d_model=opt.d_model, d_inner=opt.d_inner_hid, n_enc_layers=opt.n_enc_layers,
                                    n_dec_layers=opt.n_dec_layers, n_head=opt.n_head, d_k=opt.d_k, d_v=opt.d_v,
                                    dropout=0.1,
                                    n_src_position=100, n_trg_position=50,
                                    trg_emb_prj_weight_sharing=opt.proj_share_weight
                                    )
    checkpoint = torch.load(
        pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name, 'model', '%s.tar' % (opt.which_epoch)),
        map_location=opt.device)
    m2t_transformer.load_state_dict(checkpoint['m2t_transformer'])
    print('Loading m2t_transformer model: Epoch %03d Total_Iter %03d' % (checkpoint['ep'], checkpoint['total_it']))
    # m2t_transformer.to(opt.device)
    translator = Translator(m2t_transformer, beam_size=2, max_seq_len=30,
                            src_pad_idx=opt.mot_pad_idx, trg_pad_idx=opt.txt_pad_idx,
                            trg_sos_idx=opt.txt_start_idx, trg_eos_idx=opt.txt_end_idx)
    translator.to(opt.device)
    return translator


class M2TNMTGeneratedDataset(Dataset):

    def __init__(self, opt, dataset, w_vectorizer):
        print(opt.model_dir)

        if opt.dataset_name == 't2m':
            opt.max_motion_len = 55
            dim_pose = 263
        elif opt.dataset_name == 'kit':
            dim_pose = 251
            opt.max_motion_len = 55
        else:
            raise KeyError('Dataset Does Not Exist')

        dataloader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=True)
        opt.n_mot_vocab = opt.codebook_size + 3
        opt.mot_start_idx = opt.codebook_size
        opt.mot_end_idx = opt.codebook_size + 1
        opt.mot_pad_idx = opt.codebook_size + 2

        self.nlp = spacy.load('en_core_web_sm')


        opt.n_txt_vocab = len(w_vectorizer) + 1
        _, _, opt.txt_start_idx = w_vectorizer['sos/OTHER']
        _, _, opt.txt_end_idx = w_vectorizer['eos/OTHER']
        opt.txt_pad_idx = len(w_vectorizer)

        translator = build_models(opt)

        generated_texts_list = []
        motions_list = []
        all_captions_list = []
        m_tokens_list = []
        m_length_list = []
        t_tokens_list = []
        # print(mm_idxs)

        for i, data in tqdm(enumerate(dataloader)):
            _, _, _, _, motion, m_tokens, m_length, all_captions = data
            m_tokens = m_tokens.detach().to(opt.device).long()
            # print(m_tokens)
            # print(m_tokens.shape)
            pred_word_ids = translator.translate_sentence(m_tokens)
            pred_word_ids = pred_word_ids[1:-1]

            pred_sent = ' '.join(w_vectorizer.itos(i) for i in pred_word_ids)
            all_captions = [sentence[0] for sentence in all_captions]
            # if len(all_captions) < 3:
            #     all_captions = all_captions * 3
            # print(pred_sent)
            # print(all_captions)

            word_list, pos_list = self._process_text(pred_sent.strip())
            t_tokens = ['%s/%s' % (word_list[i], pos_list[i]) for i in range(len(word_list))]

            generated_texts_list.append(pred_sent)
            motions_list.append(motion[0].numpy())
            all_captions_list.append(all_captions)
            m_tokens_list.append(m_tokens[0].cpu().numpy())
            m_length_list.append(m_length[0].item())
            t_tokens_list.append(t_tokens)

        self.generated_texts_list = generated_texts_list
        self.t_tokens_list = t_tokens_list
        self.motion_list = motions_list
        self.all_caption_list = all_captions_list
        self.m_tokens_list = m_tokens_list
        self.m_length_list = m_length_list
        self.opt = opt
        self.w_vectorizer = w_vectorizer


    def _process_text(self, sentence):
        sentence = sentence.replace('-', '')
        doc = self.nlp(sentence)
        word_list = []
        pos_list = []
        for token in doc:
            word = token.text
            if not word.isalpha():
                continue
            if (token.pos_ == 'NOUN' or token.pos_ == 'VERB') and (word != 'left'):
                word_list.append(token.lemma_)
            else:
                word_list.append(word)
            pos_list.append(token.pos_)
        return word_list, pos_list

    def __len__(self):
        return len(self.generated_texts_list)

    def __getitem__(self, item):
        t_tokens, caption, all_captions = self.t_tokens_list[item], self.generated_texts_list[item], self.all_caption_list[item]
        motion, m_tokens, m_length = self.motion_list[item], self.m_tokens_list[item], self.m_length_list[item]

        if len(t_tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + t_tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = t_tokens[:self.opt.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh, _ = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        # print(len(word_embeddings))
        # print(len(pos_one_hots))
        # print(len(caption))
        # # print(len(caption))
        # print(len(motion))
        # print(len(m_tokens))
        # print('-------------')

        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_tokens, m_length, all_captions
