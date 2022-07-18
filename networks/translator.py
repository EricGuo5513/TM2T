import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.transformer import get_subsequent_mask, get_pad_mask_idx, get_pad_mask

class Translator(nn.Module):

    """Load a trained model and translate in beam search fashion"""

    def __init__(self, model, beam_size, max_seq_len, src_pad_idx, trg_pad_idx, trg_sos_idx, trg_eos_idx):
        super(Translator, self).__init__()
        self.alpha = 0.7
        self.beam_size=  beam_size
        self.max_seq_len = max_seq_len
        self.src_pad_idx = src_pad_idx
        self.trg_sos_idx = trg_sos_idx
        self.trg_eos_idx = trg_eos_idx

        self.model = model

        self.model.eval()
        self.register_buffer('init_seq', torch.LongTensor([[trg_sos_idx]]))
        self.register_buffer('blank_seqs',
                             torch.full((beam_size, max_seq_len), trg_pad_idx, dtype=torch.long))
        self.blank_seqs[:, 0] = self.trg_sos_idx
        self.register_buffer(
            'len_map',
            torch.arange(1, max_seq_len+1, dtype=torch.long).unsqueeze(0)
        )


    def _model_decode(self, trg_seq, enc_output, src_mask):
        trg_mask = get_subsequent_mask(trg_seq)
        dec_output, *_ = self.model.decoder(trg_seq, trg_mask, enc_output, src_mask)
        return F.softmax(self.model.trg_word_prj(dec_output), dim=-1)


    def _get_init_state(self, src_seq, src_mask):
        beam_size = self.beam_size

        enc_output, *_ = self.model.encoder(src_seq, src_mask)
        dec_output = self._model_decode(self.init_seq, enc_output, src_mask)

        best_k_probs, best_k_idx = dec_output[:, -1, :].topk(beam_size)

        scores = torch.log(best_k_probs).view(beam_size)
        gen_seq = self.blank_seqs.clone().detach()
        gen_seq[:, 1] = best_k_idx[0]
        enc_output = enc_output.repeat(beam_size, 1, 1)
        return enc_output, gen_seq, scores


    def _get_the_best_score_and_idx(self, gen_seq, dec_output, scores, step):
        assert len(scores.size()) == 1

        beam_size = self.beam_size

        best_k2_probs, best_k2_idx = dec_output[:, -1, :].topk(beam_size)

        scores = torch.log(best_k2_probs).view(beam_size, -1) + scores.view(beam_size, 1)

        scores, best_k_idx_in_k2 = scores.view(-1).topk(beam_size)

        best_k_r_idxs, best_k_c_idxs = best_k_idx_in_k2 // beam_size, best_k_idx_in_k2 % beam_size
        best_k_idx = best_k2_idx[best_k_r_idxs, best_k_c_idxs]

        gen_seq[:, :step] = gen_seq[best_k_r_idxs, :step]
        gen_seq[:, step] = best_k_idx

        return gen_seq, scores


    def translate_sentence(self, src_seq, src_non_pad_lens=None):
        assert src_seq.size(0) == 1
        src_pad_idx, trg_eos_idx = self.src_pad_idx, self.trg_eos_idx
        max_seq_len, beam_size, alpha = self.max_seq_len, self.beam_size, self.alpha

        with torch.no_grad():
            if src_non_pad_lens is None:
                src_mask = get_pad_mask_idx(src_seq, src_pad_idx)
            else:
                batch_size, src_seq_len = src_seq.shape[0], src_seq.shape[1]
                src_mask = get_pad_mask(batch_size, src_seq_len, src_non_pad_lens).to(src_seq.device)
            enc_output, gen_seq, scores = self._get_init_state(src_seq, src_mask)

            # ans_idx = 0
            for step in range(2, max_seq_len):
                dec_output = self._model_decode(gen_seq[:, :step], enc_output, src_mask)
                gen_seq, scores = self._get_the_best_score_and_idx(gen_seq, dec_output, scores, step)

                eos_locs = gen_seq == trg_eos_idx

                seq_lens, _ = self.len_map.masked_fill(~eos_locs, max_seq_len).min(1)

                if (eos_locs.sum(1)>0).sum(0).item() == beam_size:
                    break
            _, ans_idx = scores.div(seq_lens.float() ** alpha).max(0)
            ans_idx = ans_idx.item()
        return gen_seq[ans_idx][:seq_lens[ans_idx]].tolist()

    def sample(self, src_seq, sample=False, top_k=None):
        trg_seq = self.model.sample(src_seq, self.trg_sos_idx,
                                    self.trg_eos_idx, self.max_seq_len,
                                    sample, top_k)
        return trg_seq

    def sample_batch(self, src_seq, sample=False, top_k=None):
        trg_seq, len_map = self.model.sample(src_seq, self.trg_sos_idx,
                                             self.trg_eos_idx, self.max_seq_len,
                                             sample, top_k)
        return trg_seq, len_map


class Translator4Baseline(Translator):

    def _model_decode(self, src_seq, trg_input, trg_hidden):
        # trg_mask = get_subsequent_mask(trg_seq)
        # print(trg_input.shape)
        for i in range(trg_input.shape[1]):
            dec_output, trg_hidden = self.model.text_decoder_step(src_seq, trg_input[:, i], trg_hidden)
        # print(dec_output.shape)
        return F.softmax(dec_output, dim=-1)


    def _get_init_state(self, src_seq, src_non_pad_lens):
        beam_size = self.beam_size

        # print(src_seq.shape, src_non_pad_lens)
        enc_output, enc_output_last = self.model.motion_encoder(src_seq, src_non_pad_lens)
        trg_hidden = self.model.text_decoder_step.get_init_hidden(enc_output_last)

        dec_output = self._model_decode(enc_output, self.init_seq, trg_hidden)

        best_k_probs, best_k_idx = dec_output.topk(beam_size)

        scores = torch.log(best_k_probs).view(beam_size)
        gen_seq = self.blank_seqs.clone().detach()
        gen_seq[:, 1] = best_k_idx[0]
        enc_output = enc_output.repeat(beam_size, 1, 1)
        trg_hidden = [hidden.repeat(beam_size, 1) for hidden in trg_hidden]
        return enc_output, gen_seq, trg_hidden, scores


    def _get_the_best_score_and_idx(self, gen_seq, dec_step_output, scores, step):
        assert len(scores.size()) == 1

        beam_size = self.beam_size

        best_k2_probs, best_k2_idx = dec_step_output.topk(beam_size)

        scores = torch.log(best_k2_probs).view(beam_size, -1) + scores.view(beam_size, 1)

        scores, best_k_idx_in_k2 = scores.view(-1).topk(beam_size)

        best_k_r_idxs, best_k_c_idxs = best_k_idx_in_k2 // beam_size, best_k_idx_in_k2 % beam_size
        best_k_idx = best_k2_idx[best_k_r_idxs, best_k_c_idxs]

        gen_seq[:, :step] = gen_seq[best_k_r_idxs, :step]
        gen_seq[:, step] = best_k_idx

        return gen_seq, scores


    def translate_sentence(self, src_seq, src_non_pad_lens):
        assert src_seq.size(0) == 1
        src_pad_idx, trg_eos_idx = self.src_pad_idx, self.trg_eos_idx
        max_seq_len, beam_size, alpha = self.max_seq_len, self.beam_size, self.alpha

        with torch.no_grad():
            enc_output, gen_seq, trg_hidden, scores = self._get_init_state(src_seq, src_non_pad_lens)

            # ans_idx = 0
            for step in range(2, max_seq_len):
                dec_step_output = self._model_decode(enc_output, gen_seq[:, :step], trg_hidden)
                gen_seq, scores = self._get_the_best_score_and_idx(gen_seq, dec_step_output, scores, step)
                # print(gen_seq.shape)

                eos_locs = gen_seq == trg_eos_idx

                seq_lens, _ = self.len_map.masked_fill(~eos_locs, max_seq_len).min(1)

                if (eos_locs.sum(1)>0).sum(0).item() == beam_size:
                    break
            _, ans_idx = scores.div(seq_lens.float() ** alpha).max(0)
            ans_idx = ans_idx.item()
        return gen_seq[ans_idx][:seq_lens[ans_idx]].tolist()

    # def translate_sentence_uniface(self, src_seq, src_tokens, src_non_pad_lens):
    #     return self.translate_sentence(src_seq, src_non_pad_lens)