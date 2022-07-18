import torch
import torch.nn as nn
import numpy as np
from networks.layers import *


def get_pad_mask(batch_size, seq_len, non_pad_lens):
    non_pad_lens = non_pad_lens.data.tolist()
    mask_2d = torch.zeros((batch_size, seq_len), dtype=torch.float32)
    for i, cap_len in enumerate(non_pad_lens):
        mask_2d[i, :cap_len] = 1
    return mask_2d.unsqueeze(1).bool()

def get_pad_mask_idx(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(1)

def get_subsequent_mask(seq):
    sz_b, seq_len = seq.shape
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, seq_len, seq_len)), diagonal=1)).bool()
    return subsequent_mask.to(seq.device)


def cal_performance(pred, gold, trg_pad_idx, smoothing=False):
    loss = cal_loss(pred, gold, trg_pad_idx, smoothing=smoothing)
    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(trg_pad_idx)
    n_correct = pred.eq(gold).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()
    # pred = pred.masked_select(non_pad_mask)
    return loss, pred, n_correct, n_word


def cal_loss(pred, gold, trg_pad_idx, smoothing=False):
    '''Calculate cross entropy loss, apply label smoothing if needed.'''
    # gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class-1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(trg_pad_idx)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=trg_pad_idx, reduction='sum')
    return loss


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        self.encoding.requires_grad = False

        pos = torch.arange(0, max_len)
        pos = pos.float().unsqueeze(dim=1)

        _2i = torch.arange(0, d_model, step=2).float()

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        self.register_buffer('positional_encoding', self.encoding)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        return self.encoding[:seq_len, :].clone().detach().to(x.device) + x


"""Of which the inputs are vectors"""
class Encoder(nn.Module):
    def __init__(self, d_word_vec, n_layers, n_head, d_k, d_v, d_model, d_inner,
                 dropout=0.1, n_position=40):
        super(Encoder, self).__init__()
        self.position_enc = PositionalEncoding(d_model, max_len=n_position)
        self.emb = nn.Linear(d_word_vec, d_model, bias=False)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        # self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        # self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, src_seq, src_mask, return_attns=False):
        enc_slf_attn_list = []
        # print(src_seq.shape)
        src_seq = self.emb(src_seq)
        src_seq *= self.d_model ** 0.5
        enc_output = self.position_enc(src_seq)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,


"""Of which the inputs are tokens"""
class EncoderV2(nn.Module):
    def __init__(self, n_src_vocab, d_word_vec, n_layers, n_head, d_k, d_v, d_model, d_inner,
                 pad_idx, dropout=0.1, n_position=40):
        super(EncoderV2, self).__init__()
        self.position_enc = PositionalEncoding(d_model, max_len=n_position)
        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=pad_idx)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        # self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        # self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, src_seq, src_mask, return_attns=False, input_onehot=False):
        enc_slf_attn_list = []
        if input_onehot:
            src_seq = torch.matmul(src_seq, self.src_word_emb.weight)
            # print(src_seq.shape, src_mask.shape)
            src_seq = src_seq * src_mask.transpose(1, 2)
        else:
            # print(src_seq)
            src_seq = self.src_word_emb(src_seq)
        src_seq *= self.d_model ** 0.5
        enc_output = self.position_enc(src_seq)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,


"""Of which the inputs are tokens, outputs are discrete probablities"""
class Decoder(nn.Module):
    def __init__(self, n_trg_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
                 d_model, d_inner, pad_idx, n_position=200, dropout=0.1):
        super(Decoder, self).__init__()
        self.trg_word_emb = nn.Embedding(n_trg_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_word_vec, max_len=n_position)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.d_model = d_model

    def forward(self, trg_seq, trg_mask, enc_output, src_mask, return_attns=False):
        dec_slf_attn_list, dec_enc_attn_list = [], []

        dec_output = self.trg_word_emb(trg_seq)
        dec_output *= self.d_model ** 0.5

        dec_output = self.position_enc(dec_output)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask)
            dec_slf_attn_list += [dec_slf_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output,


"""Of which the inputs are vectors, outputs are discrete probablities"""
class DecoderV2(nn.Module):
    def __init__(self, d_word_vec, n_layers, n_head, d_k, d_v,
                 d_model, d_inner, n_position=200, dropout=0.1):
        super(DecoderV2, self).__init__()
        self.trg_word_emb = nn.Linear(d_word_vec, d_model, bias=False)
        self.position_enc = PositionalEncoding(d_model, max_len=n_position)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.d_model = d_model

    def forward(self, trg_seq, trg_mask, enc_output, src_mask, return_attns=False):
        dec_slf_attn_list, dec_enc_attn_list = [], []

        dec_output = self.trg_word_emb(trg_seq)
        dec_output *= self.d_model ** 0.5

        dec_output = self.position_enc(dec_output)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask)
            dec_slf_attn_list += [dec_slf_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output,


"""Of which the source sequence is vectors, and the target input is token, output is discrete probs"""
class TransformerV1(nn.Module):
    def __init__(self, n_trg_vocab, trg_pad_idx, d_src_word_vec=300, d_trg_word_vec=512,
                 d_model=512, d_inner=2048, n_enc_layers=6, n_dec_layers=6, n_head=8, d_k=64, d_v=64,
                 dropout=0.1, n_src_position=40, n_trg_position=200, trg_emb_prj_weight_sharing=True):
        super(TransformerV1, self).__init__()

        self.trg_pad_idx = trg_pad_idx

        self.d_model = d_model

        self.encoder = Encoder(
            d_word_vec=d_src_word_vec, n_position=n_src_position, d_model=d_model,
            d_inner=d_inner, n_layers=n_enc_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout
        )

        self.decoder = Decoder(
            n_trg_vocab=n_trg_vocab, n_position=n_trg_position, d_word_vec=d_trg_word_vec,
            d_model=d_model, d_inner=d_inner, n_layers=n_dec_layers, n_head=n_head, d_k=d_k,
            d_v=d_v, pad_idx=trg_pad_idx, dropout=dropout
        )
        self.trg_word_prj = nn.Linear(d_model, n_trg_vocab, bias=False)
        for p in self.parameters():
            if p.dim()>1:
                nn.init.xavier_uniform_(p)

        if trg_emb_prj_weight_sharing:
            self.trg_word_prj.weight = self.decoder.trg_word_emb.weight

    def forward(self, src_seq, trg_seq, src_non_pad_lens):
        batch_size, src_seq_len = src_seq.shape[0], src_seq.shape[1]
        src_mask = get_pad_mask(batch_size, src_seq_len, src_non_pad_lens).to(src_seq.device)
        trg_mask = get_pad_mask_idx(trg_seq, self.trg_pad_idx) & get_subsequent_mask(trg_seq)

        # print(src_mask)
        # print(trg_mask)

        enc_output, *_ = self.encoder(src_seq, src_mask)
        dec_output, *_ = self.decoder(trg_seq, trg_mask, enc_output, src_mask)
        seq_logit = self.trg_word_prj(dec_output)
        return seq_logit

    def sample(self, src_seq, src_non_pad_lens, trg_sos, trg_eos, max_steps=80, sample=False, top_k=None):
        trg_seq = torch.LongTensor(src_seq.size(0), 1).fill_(trg_sos).to(src_seq).long()

        batch_size, src_seq_len = src_seq.shape[0], src_seq.shape[1]
        src_mask = get_pad_mask(batch_size, src_seq_len, src_non_pad_lens).to(src_seq.device)
        enc_output, *_ = self.encoder(src_seq, src_mask)

        for _ in range(max_steps):
            # print(trg_seq)
            trg_mask = get_subsequent_mask(trg_seq)
            dec_output, *_ = self.decoder(trg_seq, trg_mask, enc_output, src_mask)
            seq_logit = self.trg_word_prj(dec_output)
            logits = seq_logit[:, -1, :]

            if top_k is not None:
                logits = top_k_logits(logits, top_k)
            probs = F.softmax(logits, dim=-1)
            # print(probs.sort(dim=1)[:top_k])
            # print(torch.topk(probs, k=10, dim=-1))
            _, ix = torch.topk(probs, k=1, dim=-1)
            if ix[0] == trg_eos:
                break

            if sample:
                ix = torch.multinomial(probs, num_samples=1)
                while (ix[0] in [trg_sos, trg_eos, self.trg_pad_idx]):
                    ix = torch.multinomial(probs, num_samples=1)
            trg_seq = torch.cat((trg_seq, ix), dim=1)
        return trg_seq

    def sample_batch(self, src_seq, src_non_pad_lens, trg_sos, trg_eos, max_steps=80, sample=False, top_k=None):
        trg_seq = torch.LongTensor(src_seq.size(0), 1).fill_(trg_sos).to(src_seq).long()

        batch_size, src_seq_len = src_seq.shape[0], src_seq.shape[1]
        src_mask = get_pad_mask(batch_size, src_seq_len, src_non_pad_lens).to(src_seq.device)
        enc_output, *_ = self.encoder(src_seq, src_mask)

        len_map = torch.ones((batch_size, 1), dtype=torch.long).to(src_seq.device) * max_steps

        for i in range(max_steps):
            # print(trg_seq)
            trg_mask = get_subsequent_mask(trg_seq)
            dec_output, *_ = self.decoder(trg_seq, trg_mask, enc_output, src_mask)
            seq_logit = self.trg_word_prj(dec_output)
            logits = seq_logit[:, -1, :]

            if top_k is not None:
                logits = top_k_logits(logits, top_k)
            probs = F.softmax(logits, dim=-1)
            # print(probs.sort(dim=1)[:top_k])
            # print(torch.topk(probs, k=10, dim=-1))
            _, ix = torch.topk(probs, k=1, dim=-1)

            eos_locs = (ix == trg_eos)
            non_eos_map = (len_map == max_steps)
            len_map = len_map.masked_fill(eos_locs & non_eos_map, i)

            if (len_map.ne(max_steps)).sum() == batch_size:
                break

            if sample:
                probs[:, [trg_eos, trg_sos, self.trg_pad_idx]] = 0

                ix = torch.multinomial(probs, num_samples=1)
            ix.masked_fill_(eos_locs, trg_eos)
            trg_seq = torch.cat((trg_seq, ix), dim=1)

        return trg_seq, len_map


"""Of which the source sequence is tokens, and the target input is token, output is discrete probs"""
"""Pretrained Word Embeddings are not used"""
class TransformerV2(nn.Module):
    def __init__(self, n_src_vocab, src_pad_idx, n_trg_vocab, trg_pad_idx, d_src_word_vec=512, d_trg_word_vec=512,
                 d_model=512, d_inner=2048, n_enc_layers=6, n_dec_layers=6, n_head=8, d_k=64, d_v=64,
                 dropout=0.1, n_src_position=40, n_trg_position=200, trg_emb_prj_weight_sharing=True):
        super(TransformerV2, self).__init__()

        self.trg_pad_idx = trg_pad_idx
        self.src_pad_idx = src_pad_idx

        self.d_model = d_model

        self.encoder = EncoderV2(
            n_src_vocab=n_src_vocab, n_position=n_src_position, d_word_vec=d_src_word_vec,
            d_model=d_model, d_inner=d_inner, n_layers=n_enc_layers, n_head=n_head, d_k=d_k,
            d_v=d_v,  pad_idx=src_pad_idx, dropout=dropout
        )

        self.decoder = Decoder(
            n_trg_vocab=n_trg_vocab, n_position=n_trg_position, d_word_vec=d_trg_word_vec,
            d_model=d_model, d_inner=d_inner, n_layers=n_dec_layers, n_head=n_head, d_k=d_k,
            d_v=d_v, pad_idx=trg_pad_idx, dropout=dropout
        )
        self.trg_word_prj = nn.Linear(d_model, n_trg_vocab, bias=False)
        for p in self.parameters():
            if p.dim()>1:
                nn.init.xavier_uniform_(p)

        if trg_emb_prj_weight_sharing:
            self.trg_word_prj.weight = self.decoder.trg_word_emb.weight

    def forward(self, src_seq, trg_seq, input_onehot=False, src_mask=None, src_non_pad_lens=None):
        batch_size, src_seq_len = src_seq.shape[0], src_seq.shape[1]
        # src_mask = get_pad_mask(batch_size, src_seq_len, src_non_pad_lens).to(src_seq.device)
        if not input_onehot:
            src_mask = get_pad_mask_idx(src_seq, self.src_pad_idx)
        elif src_mask is None:
            src_mask = get_pad_mask(batch_size, src_seq_len, src_non_pad_lens).to(src_seq.device)
        trg_mask = get_pad_mask_idx(trg_seq, self.trg_pad_idx) & get_subsequent_mask(trg_seq)

        # print(src_mask)
        # print(trg_mask)

        enc_output, *_ = self.encoder(src_seq, src_mask, input_onehot, input_onehot=input_onehot)
        dec_output, *_ = self.decoder(trg_seq, trg_mask, enc_output, src_mask)
        seq_logit = self.trg_word_prj(dec_output)
        return seq_logit

    def sample(self, src_seq, trg_sos, trg_eos, max_steps=80, sample=False, top_k=None):
        trg_seq = torch.LongTensor(src_seq.size(0), 1).fill_(trg_sos).to(src_seq).long()

        # batch_size, src_seq_len = src_seq.shape[0], src_seq.shape[1]
        # src_mask = get_pad_mask(batch_size, src_seq_len, src_non_pad_lens).to(src_seq.device)
        src_mask = get_pad_mask_idx(src_seq, self.src_pad_idx)
        enc_output, *_ = self.encoder(src_seq, src_mask)

        for _ in range(max_steps):
            # print(trg_seq)
            trg_mask = get_subsequent_mask(trg_seq)
            dec_output, *_ = self.decoder(trg_seq, trg_mask, enc_output, src_mask)
            seq_logit = self.trg_word_prj(dec_output)
            logits = seq_logit[:, -1, :]

            if top_k is not None:
                logits = top_k_logits(logits, top_k)
            probs = F.softmax(logits, dim=-1)
            # print(probs.sort(dim=1)[:top_k])
            # print(torch.topk(probs, k=10, dim=-1))
            _, ix = torch.topk(probs, k=1, dim=-1)
            if ix[0] == trg_eos:
                break

            if sample:
                ix = torch.multinomial(probs, num_samples=1)
                while (ix[0] in [trg_sos, trg_eos]):
                    ix = torch.multinomial(probs, num_samples=1)
            trg_seq = torch.cat((trg_seq, ix), dim=1)
        return trg_seq


    def sample_batch(self, src_seq, trg_sos, trg_eos, max_steps=80, sample=False, top_k=None):
        trg_seq = torch.LongTensor(src_seq.size(0), 1).fill_(trg_sos).to(src_seq).long()

        batch_size, src_seq_len = src_seq.shape[0], src_seq.shape[1]
        src_mask = get_pad_mask_idx(src_seq, self.src_pad_idx)
        enc_output, *_ = self.encoder(src_seq, src_mask)

        len_map = torch.ones((batch_size, 1), dtype=torch.long).to(src_seq.device) * max_steps

        for i in range(max_steps):
            # print(trg_seq)
            trg_mask = get_subsequent_mask(trg_seq)
            dec_output, *_ = self.decoder(trg_seq, trg_mask, enc_output, src_mask)
            seq_logit = self.trg_word_prj(dec_output)
            logits = seq_logit[:, -1, :]

            if top_k is not None:
                logits = top_k_logits(logits, top_k)
            probs = F.softmax(logits, dim=-1)
            # print(probs.sort(dim=1)[:top_k])
            # print(torch.topk(probs, k=10, dim=-1))
            _, ix = torch.topk(probs, k=1, dim=-1)

            eos_locs = (ix == trg_eos)
            non_eos_map = (len_map == max_steps)
            len_map = len_map.masked_fill(eos_locs & non_eos_map, i)

            if (len_map.ne(max_steps)).sum() == batch_size:
                break

            if sample:
                probs[:, [trg_eos, trg_sos, self.trg_pad_idx]] = 0

                ix = torch.multinomial(probs, num_samples=1)
            ix.masked_fill_(eos_locs, trg_eos)
            trg_seq = torch.cat((trg_seq, ix), dim=1)

        return trg_seq, len_map


"""Of which the source sequence is tokens, and the target input is vectors, output is discrete probs"""
"""Pretrained Word Embeddings are used"""
class TransformerV3(nn.Module):
    def __init__(self, n_src_vocab, src_pad_idx, n_trg_vocab, trg_pad_idx, d_src_word_vec=512, d_trg_word_vec=512,
                 d_model=512, d_inner=2048, n_enc_layers=6, n_dec_layers=6, n_head=8, d_k=64, d_v=64,
                 dropout=0.1, n_src_position=40, n_trg_position=200):
        super(TransformerV3, self).__init__()

        self.trg_pad_idx = trg_pad_idx
        self.src_pad_idx = src_pad_idx

        self.d_model = d_model

        self.encoder = EncoderV2(
            n_src_vocab=n_src_vocab, n_position=n_src_position, d_word_vec=d_src_word_vec,
            d_model=d_model, d_inner=d_inner, n_layers=n_enc_layers, n_head=n_head, d_k=d_k,
            d_v=d_v,  pad_idx=src_pad_idx, dropout=dropout
        )

        self.decoder = DecoderV2(
            n_position=n_trg_position, d_word_vec=d_trg_word_vec,
            d_model=d_model, d_inner=d_inner, n_layers=n_dec_layers, n_head=n_head, d_k=d_k,
            d_v=d_v, dropout=dropout
        )

        self.trg_word_prj = nn.Linear(d_model, n_trg_vocab, bias=False)
        for p in self.parameters():
            if p.dim()>1:
                nn.init.xavier_uniform_(p)

        # if trg_emb_prj_weight_sharing:
        #     self.trg_word_prj.weight = self.decoder.trg_word_emb.weight

    def forward(self, src_seq, trg_seq, trg_non_pad_lens):
        batch_size, trg_seq_len = trg_seq.shape[0], trg_seq.shape[1]
        # src_mask = get_pad_mask(batch_size, src_seq_len, src_non_pad_lens).to(src_seq.device)
        src_mask = get_pad_mask_idx(src_seq, self.src_pad_idx)
        trg_mask = get_pad_mask(batch_size, trg_seq_len, trg_non_pad_lens).to(src_seq.device) & get_subsequent_mask(trg_seq[:, :, 0])

        enc_output, *_ = self.encoder(src_seq, src_mask)
        dec_output, *_ = self.decoder(trg_seq, trg_mask, enc_output, src_mask)
        seq_logit = self.trg_word_prj(dec_output)
        return seq_logit

    def sample(self, src_seq, trg_sos, trg_eos, max_steps=80, sample=False, top_k=None):
        trg_seq = torch.LongTensor(src_seq.size(0), 1).fill_(trg_sos).to(src_seq).long()

        # batch_size, src_seq_len = src_seq.shape[0], src_seq.shape[1]
        # src_mask = get_pad_mask(batch_size, src_seq_len, src_non_pad_lens).to(src_seq.device)
        src_mask = get_pad_mask_idx(src_seq, self.src_pad_idx)
        enc_output, *_ = self.encoder(src_seq, src_mask)

        for _ in range(max_steps):
            # print(trg_seq)
            trg_mask = get_subsequent_mask(trg_seq)
            dec_output, *_ = self.decoder(trg_seq, trg_mask, enc_output, src_mask)
            seq_logit = self.trg_word_prj(dec_output)
            logits = seq_logit[:, -1, :]

            if top_k is not None:
                logits = top_k_logits(logits, top_k)
            probs = F.softmax(logits, dim=-1)
            # print(probs.sort(dim=1)[:top_k])
            # print(torch.topk(probs, k=10, dim=-1))
            _, ix = torch.topk(probs, k=1, dim=-1)
            if ix[0] == trg_eos:
                break

            if sample:
                ix = torch.multinomial(probs, num_samples=1)
                while (ix[0] in [trg_sos, trg_eos]):
                    ix = torch.multinomial(probs, num_samples=1)
            trg_seq = torch.cat((trg_seq, ix), dim=1)
        return trg_seq


def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out
