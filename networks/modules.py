import torch
import torch.nn as nn
import numpy as np
import time
import math
import random
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from networks.layers import *


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    def __init__(self, margin=3.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive


def init_weight(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose1d):
        nn.init.xavier_normal_(m.weight)
        # m.bias.data.fill_(0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


# batch_size, dimension and position
# output: (batch_size, dim)
def positional_encoding(batch_size, dim, pos):
    assert batch_size == pos.shape[0]
    positions_enc = np.array([
        [pos[j] / np.power(10000, (i-i%2)/dim) for i in range(dim)]
        for j in range(batch_size)
    ], dtype=np.float32)
    positions_enc[:, 0::2] = np.sin(positions_enc[:, 0::2])
    positions_enc[:, 1::2] = np.cos(positions_enc[:, 1::2])
    return torch.from_numpy(positions_enc).float()


def get_padding_mask(batch_size, seq_len, cap_lens):
    cap_lens = cap_lens.data.tolist()
    mask_2d = torch.ones((batch_size, seq_len, seq_len), dtype=torch.float32)
    for i, cap_len in enumerate(cap_lens):
        mask_2d[i, :, :cap_len] = 0
    return mask_2d.bool(), 1 - mask_2d[:, :, 0].clone()


def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=300):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, pos):
        return self.pe[pos]


class MovementConvEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MovementConvEncoder, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, 4, 2, 1),
            nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden_size, output_size, 4, 2, 1),
            nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.out_net = nn.Linear(output_size, output_size)
        self.main.apply(init_weight)
        self.out_net.apply(init_weight)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        outputs = self.main(inputs).permute(0, 2, 1)
        # print(outputs.shape)
        return self.out_net(outputs)


class MovementConvDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MovementConvDecoder, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose1d(input_size, hidden_size, 4, 2, 1),
            # nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(hidden_size, output_size, 4, 2, 1),
            # nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.out_net = nn.Linear(output_size, output_size)

        self.main.apply(init_weight)
        self.out_net.apply(init_weight)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        outputs = self.main(inputs).permute(0, 2, 1)
        return self.out_net(outputs)


class VQEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(VQEncoder, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, 4, 2, 1),
            # nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden_size, output_size, 4, 2, 1),
            # nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.out_net = nn.Linear(output_size, output_size)
        self.main.apply(init_weight)
        self.out_net.apply(init_weight)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        outputs = self.main(inputs).permute(0, 2, 1)
        # print(outputs.shape)
        return self.out_net(outputs)


class VQEncoderV2(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(VQEncoderV2, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(input_size, hidden_size//2, 4, 2, 1),
            # nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden_size//2, hidden_size, 4, 2, 1),
            # nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden_size, hidden_size, 4, 2, 1),
            # nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden_size, output_size, 3, 1, 1),
        )
        # self.out_net = nn.Linear(output_size, output_size)
        self.main.apply(init_weight)
        # self.out_net.apply(init_weight)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        outputs = self.main(inputs).permute(0, 2, 1)
        # print(outputs.shape)
        return outputs


class VQEncoderV3(nn.Module):
    def __init__(self, input_size, channels, n_down):
        super(VQEncoderV3, self).__init__()
        assert len(channels) == n_down
        layers = [
            nn.Conv1d(input_size, channels[0], 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(channels[0]),
        ]

        for i in range(1, n_down):
            layers += [
                nn.Conv1d(channels[i-1], channels[i], 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                ResBlock(channels[i]),
            ]
        self.main = nn.Sequential(*layers)
        # self.out_net = nn.Linear(output_size, output_size)
        self.main.apply(init_weight)
        # self.out_net.apply(init_weight)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        outputs = self.main(inputs).permute(0, 2, 1)
        # print(outputs.shape)
        return outputs


class VQDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(VQDecoder, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose1d(input_size, hidden_size, 4, 2, 1),
            # nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(hidden_size, output_size, 4, 2, 1),
            # nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.out_net = nn.Linear(output_size, output_size)

        self.main.apply(init_weight)
        self.out_net.apply(init_weight)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        outputs = self.main(inputs).permute(0, 2, 1)
        return self.out_net(outputs)


class VQDecoderV2(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(VQDecoderV2, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose1d(input_size, hidden_size//2, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(hidden_size//2, hidden_size, 4, 2, 1),
            # nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(hidden_size, hidden_size, 4, 2, 1),
            # nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(hidden_size, output_size, 4, 2, 1),
            # nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.out_net = nn.Linear(output_size, output_size)

        self.main.apply(init_weight)
        self.out_net.apply(init_weight)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        outputs = self.main(inputs).permute(0, 2, 1)
        return self.out_net(outputs)


class VQDecoderV3(nn.Module):
    def __init__(self, input_size, channels, n_resblk, n_up):
        super(VQDecoderV3, self).__init__()
        assert len(channels) == n_up + 1
        if input_size == channels[0]:
            layers = []
        else:
            layers = [nn.Conv1d(input_size, channels[0], kernel_size=3, stride=1, padding=1)]

        for i in range(n_resblk):
            layers += [ResBlock(channels[0])]
        # channels = channels
        for i in range(n_up):
            layers += [
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(channels[i], channels[i+1], kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        layers += [nn.Conv1d(channels[-1], channels[-1], kernel_size=3, stride=1, padding=1)]
        self.main = nn.Sequential(*layers)
        self.main.apply(init_weight)
        # self.out_net.apply(init_weight)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        outputs = self.main(inputs).permute(0, 2, 1)
        return outputs


class VQDiscriminator(nn.Module):
    def __init__(self, input_size, hidden_size, n_layer):
        super(VQDiscriminator, self).__init__()
        sequence = [nn.Conv1d(input_size, hidden_size, 4, 2, 1),
                    nn.BatchNorm1d(hidden_size),
                    nn.LeakyReLU(0.2, inplace=True)
                    ]
        layer_size = hidden_size
        for i in range(n_layer-1):
            sequence += [
                    nn.Conv1d(layer_size, layer_size//2, 4, 2, 1),
                    nn.BatchNorm1d(layer_size//2),
                    nn.LeakyReLU(0.2, inplace=True)
            ]
            layer_size = layer_size // 2

        self.out_net = nn.Conv1d(layer_size, 1, 3, 1, 1)
        self.main = nn.Sequential(*sequence)

        self.out_net.apply(init_weight)
        self.main.apply(init_weight)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        feats = self.main(inputs)
        outs = self.out_net(feats)
        return feats.permute(0, 2, 1), outs.permute(0, 2, 1)


class TextEncoderBiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, device):
        super(TextEncoderBiGRU, self).__init__()
        self.device = device

        self.input_emb = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.input_emb.apply(init_weight)
        self.hidden_size = hidden_size
        self.hidden = nn.Parameter(torch.randn((2, 1, self.hidden_size), requires_grad=True))

    def forward(self, inputs, m_lens):
        num_samples = inputs.shape[0]

        input_embs = self.input_emb(inputs)
        hidden = self.hidden.repeat(1, num_samples, 1)

        cap_lens = m_lens.data.tolist()
        emb = pack_padded_sequence(input_embs, cap_lens, batch_first=True)

        # print(input_embs.shape)
        # print(cap_lens)
        # print(input_embs.shape, hidden.shape)
        gru_seq, gru_last = self.gru(emb, hidden)

        gru_last = torch.cat([gru_last[0], gru_last[1]], dim=-1)

        # print(gru_seq.shape)
        # print(self.hidden_size)
        gru_seq = pad_packed_sequence(gru_seq, batch_first=True)[0]

        forward_seq = gru_seq[..., :self.hidden_size]
        backward_seq = gru_seq[..., self.hidden_size:].clone()

        # Concate the forward and backward word embeddings
        for i, length in enumerate(m_lens):
            backward_seq[i:i + 1, :length] = torch.flip(backward_seq[i:i + 1, :length].clone(), dims=[1])
        gru_seq = torch.cat([forward_seq, backward_seq], dim=-1)

        return gru_seq, gru_last


class MotionLateAttDecoder(nn.Module):
    def __init__(self, n_mot_vocab, init_hidden_size, hidden_size, n_layers, device):
        super(MotionLateAttDecoder, self).__init__()
        self.device = device

        self.input_emb = nn.Embedding(n_mot_vocab, hidden_size)
        self.n_layers = n_layers

        self.z2init = nn.Linear(init_hidden_size, hidden_size * n_layers)
        self.gru = nn.ModuleList([nn.GRUCell(hidden_size, hidden_size) for i in range(self.n_layers)])

        self.att_layer = AttLayer(hidden_size, init_hidden_size, hidden_size)
        self.att_linear = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, hidden_size)
        )
        self.trg_word_prj = nn.Linear(hidden_size, n_mot_vocab, bias=False)

        self.input_emb.apply(init_weight)
        self.z2init.apply(init_weight)
        # self.output_net.apply(init_weight)
        self.att_layer.apply(init_weight)
        self.att_linear.apply(init_weight)
        self.hidden_size = hidden_size

        self.trg_word_prj.weight = self.input_emb.weight

    def get_init_hidden(self, latent):
        hidden = self.z2init(latent)
        hidden = torch.split(hidden, self.hidden_size, dim=-1)
        return list(hidden)

    # input(batch_size, dim)
    def forward(self, src_output, inputs, hidden):
        h_in = self.input_emb(inputs)

        # h_in *= self.hidden_size ** 0.5

        for i in range(self.n_layers):
            # print(h_in.shape, hidden[i].shape)
            hidden[i] = self.gru[i](h_in, hidden[i])
            h_in = hidden[i]

        # print(h_in.shape, src_output.shape)
        att_vec, _ = self.att_layer(h_in, src_output)
        pred_probs = self.trg_word_prj(
            self.att_linear(
                torch.cat([att_vec, h_in], dim=-1)
            )
        )
        return pred_probs, hidden


class MotionEarlyAttDecoder(nn.Module):
    def __init__(self, n_mot_vocab, init_hidden_size, hidden_size, n_layers, device):
        super(MotionEarlyAttDecoder, self).__init__()
        self.device = device

        self.input_emb = nn.Embedding(n_mot_vocab, hidden_size)
        self.n_layers = n_layers

        self.z2init = nn.Linear(init_hidden_size, hidden_size * n_layers)
        self.gru = nn.ModuleList([nn.GRUCell(hidden_size, hidden_size) for i in range(self.n_layers)])

        self.att_layer = AttLayer(hidden_size, init_hidden_size, hidden_size)
        self.att_linear = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.trg_word_prj = nn.Linear(hidden_size, n_mot_vocab, bias=False)

        self.input_emb.apply(init_weight)
        self.z2init.apply(init_weight)
        # self.output_net.apply(init_weight)
        self.att_layer.apply(init_weight)
        self.att_linear.apply(init_weight)
        self.hidden_size = hidden_size

        self.trg_word_prj.weight = self.input_emb.weight

    def get_init_hidden(self, latent):
        hidden = self.z2init(latent)
        hidden = torch.split(hidden, self.hidden_size, dim=-1)
        return list(hidden)

    # input(batch_size, dim)
    def forward(self, src_output, inputs, hidden):
        h_in = self.input_emb(inputs)

        # h_in *= self.hidden_size ** 0.5

        att_vec, _ = self.att_layer(hidden[-1], src_output)
        # print(att_vec.shape, h_in.shape)
        h_in = self.att_linear(
            torch.cat([att_vec, h_in], dim=-1)
        )

        for i in range(self.n_layers):
            # print(h_in.shape, hidden[i].shape)
            hidden[i] = self.gru[i](h_in, hidden[i])
            h_in = hidden[i]

        # print(h_in.shape, src_output.shape)
        pred_probs = self.trg_word_prj(h_in)
        return pred_probs, hidden


class Seq2SeqText2MotModel(nn.Module):
    def __init__(self, dim_txt, n_mot_vocab, dim_txt_hid, dim_mot_hid, n_mot_layers, device, early_or_late="early"):
        super(Seq2SeqText2MotModel, self).__init__()
        # dim_init_txt_hid = dim_txt_hid
        self.n_mot_vocab = n_mot_vocab
        self.text_encoder = TextEncoderBiGRU(dim_txt, dim_txt_hid, device)
        if early_or_late == "early":
            self.motion_decoder_step = MotionEarlyAttDecoder(n_mot_vocab, dim_txt_hid*2, dim_mot_hid, n_mot_layers, device)
        elif early_or_late == "late":
            self.motion_decoder_step = MotionLateAttDecoder(n_mot_vocab, dim_txt_hid*2, dim_mot_hid, n_mot_layers, device)

    def forward(self, src_seq, trg_input, src_non_pad_lens, tf_ratio):
        txt_hid_seq, txt_hid_last = self.text_encoder(src_seq, src_non_pad_lens)
        motion_hidden = self.motion_decoder_step.get_init_hidden(txt_hid_last)

        flipped_coin = random.random()

        is_tf = True if flipped_coin < tf_ratio else False

        trg_len = trg_input.shape[1]
        trg_step_input = trg_input[:, 0]
        trg_pred_output = []
        for i in range(0, trg_len):
            trg_step_pred, text_hidden = self.motion_decoder_step(txt_hid_seq, trg_step_input.detach(), motion_hidden)
            trg_pred_output.append(trg_step_pred.unsqueeze(1))
            if is_tf and i+1 < trg_len:
                trg_step_input = trg_input[:, i+1]
            else:
                _, trg_step_input = trg_step_pred.max(-1)

        trg_pred_output = torch.cat(trg_pred_output, dim=1)

        return trg_pred_output


    def gumbel_sample(self, src_seq, src_non_pad_lens, trg_sos, trg_eos, max_steps=55, top_k=-1):

        batch_size, src_seq_len = src_seq.shape[0], src_seq.shape[1]
        txt_hid_seq, txt_hid_last = self.text_encoder(src_seq, src_non_pad_lens)
        motion_hidden = self.motion_decoder_step.get_init_hidden(txt_hid_last)
        len_map = torch.ones((batch_size, 1), dtype=torch.long).to(src_seq.device) * max_steps

        trg_step_input = torch.LongTensor(src_seq.size(0)).fill_(trg_sos).to(src_seq).long()
        trg_pred_output = [F.one_hot(trg_step_input, num_classes=self.n_mot_vocab)]
        for i in range(max_steps):
            trg_step_pred, text_hidden = self.motion_decoder_step(txt_hid_seq, trg_step_input.detach(), motion_hidden)
            logits = trg_step_pred
            if top_k != -1:
                logits = top_k_logits(logits, top_k)
            # probs = F.softmax(logits, dim=-1)
            # print(probs.sort(dim=1)[:top_k])
            # print(torch.topk(probs, k=10, dim=-1))
            _, ix = torch.topk(logits, k=1, dim=-1)
            eos_locs = (ix == trg_eos)
            non_eos_map = (len_map == max_steps)
            len_map = len_map.masked_fill(eos_locs & non_eos_map, i)
            if (len_map.ne(max_steps)).sum() == batch_size:
                break

            logits[:, [trg_eos, trg_sos]] = -float('Inf')

            # (bs, num_class)
            pred_ohot = F.gumbel_softmax(logits, tau=1.0, hard=True)
            trg_pred_output.append(pred_ohot)

            # ()
            _, trg_step_input = pred_ohot.max(-1)
        trg_pred_output = torch.cat([element.unsqueeze(1) for element in trg_pred_output], dim=1)

        return trg_pred_output, len_map


    def sample_batch(self, src_seq, src_non_pad_lens, trg_sos, trg_eos, max_steps=80, top_k=None):

        batch_size, src_seq_len = src_seq.shape[0], src_seq.shape[1]
        txt_hid_seq, txt_hid_last = self.text_encoder(src_seq, src_non_pad_lens)
        motion_hidden = self.motion_decoder_step.get_init_hidden(txt_hid_last)
        len_map = torch.ones((batch_size, 1), dtype=torch.long).to(src_seq.device) * max_steps

        trg_step_input = torch.LongTensor(src_seq.size(0)).fill_(trg_sos).to(src_seq).long()
        trg_seq = [trg_step_input.unsqueeze(-1)]
        for i in range(max_steps):
            trg_step_pred, text_hidden = self.motion_decoder_step(txt_hid_seq, trg_step_input.detach(), motion_hidden)
            logits = trg_step_pred
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

            # probs[:, [trg_eos, trg_sos, self.trg_pad_idx]] = 0

            # (bs, num_class)
            probs[:, [trg_eos, trg_sos, 1026]] = 0

            ix = torch.multinomial(probs, num_samples=1)
            ix.masked_fill_(eos_locs, trg_eos)
            trg_seq.append(ix)
            trg_step_input = ix.squeeze(-1)
            # print(trg_step_input.shape)
        trg_seq = torch.cat(trg_seq, dim=-1)
        return trg_seq, len_map


#############
# For implementation of Baseline2018
#############
class MotionEncoderBiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, device):
        super(MotionEncoderBiGRU, self).__init__()
        self.device = device

        self.input_emb = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        # self.output_net = nn.Sequential(
        #     nn.Linear(hidden_size*2, hidden_size),
        #     nn.LayerNorm(hidden_size),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Linear(hidden_size, output_size)
        # )

        self.input_emb.apply(init_weight)
        # self.output_net.apply(init_weight)
        self.hidden_size = hidden_size
        self.hidden = nn.Parameter(torch.randn((2, 1, self.hidden_size), requires_grad=True))

    # input(batch_size, seq_len, dim)
    def forward(self, inputs, m_lens):
        num_samples = inputs.shape[0]

        input_embs = self.input_emb(inputs)
        hidden = self.hidden.repeat(1, num_samples, 1)

        cap_lens = m_lens.data.tolist()
        emb = pack_padded_sequence(input_embs, cap_lens, batch_first=True)

        # print(input_embs.shape)
        # print(cap_lens)
        # print(input_embs.shape, hidden.shape)
        gru_seq, gru_last = self.gru(emb, hidden)

        gru_last = torch.cat([gru_last[0], gru_last[1]], dim=-1)

        # print(gru_seq.shape)
        # print(self.hidden_size)
        gru_seq = pad_packed_sequence(gru_seq, batch_first=True)[0]

        forward_seq = gru_seq[..., :self.hidden_size]
        backward_seq = gru_seq[..., self.hidden_size:].clone()

        # Concate the forward and backward word embeddings
        for i, length in enumerate(m_lens):
            backward_seq[i:i + 1, :length] = torch.flip(backward_seq[i:i + 1, :length].clone(), dims=[1])
        gru_seq = torch.cat([forward_seq, backward_seq], dim=-1)

        return gru_seq, gru_last


class TextDecoder(nn.Module):
    def __init__(self, n_txt_vocab, init_hidden_size, hidden_size, n_layers, device):
        super(TextDecoder, self).__init__()
        self.device = device

        self.input_emb = nn.Embedding(n_txt_vocab, hidden_size)
        self.n_layers = n_layers

        self.z2init = nn.Linear(init_hidden_size, hidden_size * n_layers)
        self.gru = nn.ModuleList([nn.GRUCell(hidden_size, hidden_size) for i in range(self.n_layers)])

        self.trg_word_prj = nn.Linear(hidden_size, n_txt_vocab, bias=False)

        self.input_emb.apply(init_weight)
        self.z2init.apply(init_weight)
        # self.output_net.apply(init_weight)
        self.hidden_size = hidden_size

        self.trg_word_prj.weight = self.input_emb.weight

    def get_init_hidden(self, latent):
        hidden = self.z2init(latent)
        hidden = torch.split(hidden, self.hidden_size, dim=-1)
        return list(hidden)

    # input(batch_size, dim)
    def forward(self, src_output, inputs, hidden):
        h_in = self.input_emb(inputs)

        for i in range(self.n_layers):
            # print(h_in.shape)
            hidden[i] = self.gru[i](h_in, hidden[i])
            h_in = hidden[i]

        pred_probs = self.trg_word_prj(h_in)
        return pred_probs, hidden


class AttLayer(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim):
        super(AttLayer, self).__init__()
        self.W_q = nn.Linear(query_dim, value_dim)
        self.W_k = nn.Linear(key_dim, value_dim, bias=False)
        self.W_v = nn.Linear(key_dim, value_dim)

        self.softmax = nn.Softmax(dim=1)
        self.dim = value_dim

        self.W_q.apply(init_weight)
        self.W_k.apply(init_weight)
        self.W_v.apply(init_weight)

    def forward(self, query, key_mat):
        '''
        query (batch, query_dim)
        key (batch, seq_len, key_dim)
        '''
        # print(query.shape)
        query_vec = self.W_q(query).unsqueeze(-1)       # (batch, value_dim, 1)
        val_set = self.W_v(key_mat)                     # (batch, seq_len, value_dim)
        key_set = self.W_k(key_mat)                     # (batch, seq_len, value_dim)

        weights = torch.matmul(key_set, query_vec) / np.sqrt(self.dim)

        co_weights = self.softmax(weights)              # (batch, seq_len, 1)
        values = val_set * co_weights                   # (batch, seq_len, value_dim)
        pred = values.sum(dim=1)                        # (batch, value_dim)
        return pred, co_weights


class TextAttentionDecoder(nn.Module):
    def __init__(self, n_txt_vocab, init_hidden_size, hidden_size, n_layers, device):
        super(TextAttentionDecoder, self).__init__()
        self.device = device

        self.input_emb = nn.Embedding(n_txt_vocab, hidden_size)
        self.n_layers = n_layers

        self.z2init = nn.Linear(init_hidden_size, hidden_size * n_layers)
        self.gru = nn.ModuleList([nn.GRUCell(hidden_size, hidden_size) for i in range(self.n_layers)])

        self.att_layer = AttLayer(hidden_size, hidden_size*2, hidden_size)
        self.att_linear = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, hidden_size)
        )
        self.trg_word_prj = nn.Linear(hidden_size, n_txt_vocab, bias=False)

        self.input_emb.apply(init_weight)
        self.z2init.apply(init_weight)
        # self.output_net.apply(init_weight)
        self.att_layer.apply(init_weight)
        self.att_linear.apply(init_weight)
        self.hidden_size = hidden_size

        self.trg_word_prj.weight = self.input_emb.weight

    def get_init_hidden(self, latent):
        hidden = self.z2init(latent)
        hidden = torch.split(hidden, self.hidden_size, dim=-1)
        return list(hidden)

    # input(batch_size, dim)
    def forward(self, src_output, inputs, hidden):
        h_in = self.input_emb(inputs)

        for i in range(self.n_layers):
            # print(h_in.shape, hidden[i].shape)
            hidden[i] = self.gru[i](h_in, hidden[i])
            h_in = hidden[i]

        # print(h_in.shape, src_output.shape)
        att_vec, _ = self.att_layer(h_in, src_output)
        pred_probs = self.trg_word_prj(
            self.att_linear(
                torch.cat([att_vec, h_in], dim=-1)
            )
        )
        return pred_probs, hidden


class Seq2SeqMot2TextModel(nn.Module):
    def __init__(self, dim_pose, n_txt_vocab, dim_mot_hid, dim_txt_hid, n_txt_layers, device, use_att=False):
        super(Seq2SeqMot2TextModel, self).__init__()
        # dim_init_txt_hid = dim_txt_hid
        self.motion_encoder = MotionEncoderBiGRU(dim_pose, dim_mot_hid, device)
        if use_att:
            self.text_decoder_step = TextAttentionDecoder(n_txt_vocab, dim_mot_hid*2, dim_txt_hid, n_txt_layers, device)
        else:
            self.text_decoder_step = TextDecoder(n_txt_vocab, dim_mot_hid*2, dim_txt_hid, n_txt_layers, device)

    def forward(self, src_seq, trg_input, src_non_pad_lens, tf_ratio):
        mot_hid_seq, mot_hid_last = self.motion_encoder(src_seq, src_non_pad_lens)
        text_hidden = self.text_decoder_step.get_init_hidden(mot_hid_last)

        flipped_coin = random.random()

        is_tf = True if flipped_coin < tf_ratio else False

        trg_len = trg_input.shape[1]
        trg_step_input = trg_input[:, 0]
        trg_pred_output = []
        for i in range(0, trg_len):
            trg_step_pred, text_hidden = self.text_decoder_step(mot_hid_seq, trg_step_input.detach(), text_hidden)
            trg_pred_output.append(trg_step_pred.unsqueeze(1))
            if is_tf and i+1 < trg_len:
                trg_step_input = trg_input[:, i+1]
            else:
                _, trg_step_input = trg_step_pred.max(-1)

        trg_pred_output = torch.cat(trg_pred_output, dim=1)

        return trg_pred_output
        # self.


class TextDiscriminator(nn.Module):

    def __init__(self, n_txt_vocab, emb_dim, dim_channel, kernel_wins, dropout_rate):
        super(TextDiscriminator, self).__init__()

        self.embed = nn.Linear(n_txt_vocab, emb_dim)

        # Convolutional Layers with different window size kernels
        self.convs = nn.ModuleList([nn.Conv2d(1, dim_channel, (w, emb_dim)) for w in kernel_wins])
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)

        # FC layer
        self.fc = nn.Linear(len(kernel_wins) * dim_channel, 1)

    def forward(self, x):
        emb_x = self.embed(x)
        emb_x = emb_x.unsqueeze(1)

        con_x = [conv(emb_x) for conv in self.convs]

        pool_x = [F.max_pool1d(x.squeeze(-1), x.size()[2]) for x in con_x]

        fc_x = torch.cat(pool_x, dim=1)

        fc_x = fc_x.squeeze(-1)

        fc_x = self.dropout(fc_x)
        logit = self.fc(fc_x)
        return logit


    # class EmbeddingEMA(nn.Module)

class TextVAEDecoder(nn.Module):
    def __init__(self, text_size, input_size, output_size, hidden_size, n_layers):
        super(TextVAEDecoder, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.emb = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2, inplace=True))

        self.z2init = nn.Linear(text_size, hidden_size * n_layers)
        self.gru = nn.ModuleList([nn.GRUCell(hidden_size, hidden_size) for i in range(self.n_layers)])
        self.positional_encoder = PositionalEncoding(hidden_size)


        self.output = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, output_size)
        )

        #
        # self.output = nn.Sequential(
        #     nn.Linear(hidden_size, hidden_size),
        #     nn.LayerNorm(hidden_size),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Linear(hidden_size, output_size-4)
        # )

        # self.contact_net = nn.Sequential(
        #     nn.Linear(output_size-4, 64),
        #     nn.LayerNorm(64),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Linear(64, 4)
        # )

        self.output.apply(init_weight)
        self.emb.apply(init_weight)
        self.z2init.apply(init_weight)
        # self.contact_net.apply(init_weight)

    def get_init_hidden(self, latent):
        hidden = self.z2init(latent)
        hidden = torch.split(hidden, self.hidden_size, dim=-1)
        return list(hidden)

    def forward(self, inputs, last_pred, hidden, p):
        h_in = self.emb(inputs)
        pos_enc = self.positional_encoder(p).to(inputs.device).detach()
        h_in = h_in + pos_enc
        for i in range(self.n_layers):
            # print(h_in.shape)
            hidden[i] = self.gru[i](h_in, hidden[i])
            h_in = hidden[i]
        pose_pred = self.output(h_in)
        # pose_pred = self.output(h_in) + last_pred.detach()
        # contact = self.contact_net(pose_pred)
        # return torch.cat([pose_pred, contact], dim=-1), hidden
        return pose_pred, hidden


class TextEncoderBiGRUCo(nn.Module):
    def __init__(self, word_size, pos_size, hidden_size, output_size, device):
        super(TextEncoderBiGRUCo, self).__init__()
        self.device = device

        self.pos_emb = nn.Linear(pos_size, word_size)
        self.input_emb = nn.Linear(word_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.output_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, output_size)
        )

        self.input_emb.apply(init_weight)
        self.pos_emb.apply(init_weight)
        self.output_net.apply(init_weight)
        # self.linear2.apply(init_weight)
        # self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.hidden = nn.Parameter(torch.randn((2, 1, self.hidden_size), requires_grad=True))

    # input(batch_size, seq_len, dim)
    def forward(self, word_embs, pos_onehot, cap_lens):
        num_samples = word_embs.shape[0]

        pos_embs = self.pos_emb(pos_onehot)
        inputs = word_embs + pos_embs
        input_embs = self.input_emb(inputs)
        hidden = self.hidden.repeat(1, num_samples, 1)

        cap_lens = cap_lens.data.tolist()
        emb = pack_padded_sequence(input_embs, cap_lens, batch_first=True)

        gru_seq, gru_last = self.gru(emb, hidden)

        gru_last = torch.cat([gru_last[0], gru_last[1]], dim=-1)

        return self.output_net(gru_last)


class MotionEncoderBiGRUCo(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device):
        super(MotionEncoderBiGRUCo, self).__init__()
        self.device = device

        self.input_emb = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.output_net = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, output_size)
        )

        self.input_emb.apply(init_weight)
        self.output_net.apply(init_weight)
        self.hidden_size = hidden_size
        self.hidden = nn.Parameter(torch.randn((2, 1, self.hidden_size), requires_grad=True))

    # input(batch_size, seq_len, dim)
    def forward(self, inputs, m_lens):
        num_samples = inputs.shape[0]

        input_embs = self.input_emb(inputs)
        hidden = self.hidden.repeat(1, num_samples, 1)

        cap_lens = m_lens.data.tolist()
        emb = pack_padded_sequence(input_embs, cap_lens, batch_first=True)

        gru_seq, gru_last = self.gru(emb, hidden)

        gru_last = torch.cat([gru_last[0], gru_last[1]], dim=-1)

        return self.output_net(gru_last)


class TextEncoderTransformer(nn.Module):
    def __init__(self, word_size, pos_size, d_model, d_inner=2048, n_head=8, d_k=64, d_v=64, n_layers=6, dropout=0.1):
        super(TextEncoderTransformer, self).__init__()

        self.pos_emb = nn.Linear(pos_size, word_size)
        self.input_emb = nn.Linear(word_size, d_model)
        self.positional_encoder = PositionalEncoding(d_model, max_len=25)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)
        ])

        self.input_emb.apply(init_weight)
        self.pos_emb.apply(init_weight)

    # input(batch_size, seq_len, dim)
    def forward(self, word_embs, pos_onehot, cap_lens):
        num_samples, max_len, _ = word_embs.shape

        pos_embs = self.pos_emb(pos_onehot)
        inputs = word_embs + pos_embs
        input_embs = self.input_emb(inputs)

        pos_enc = self.positional_encoder([i for i in range(max_len)]).detach().to(input_embs.device)
        enc_output = input_embs + pos_enc

        slf_attn_mask, non_pad_mask = get_padding_mask(num_samples, max_len, cap_lens)
        slf_attn_mask = slf_attn_mask.to(input_embs.device)
        non_pad_mask = non_pad_mask.unsqueeze(-1).to(input_embs.device)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask, non_pad_mask)

        seq_vecs, word_vecs = enc_output[:, 0].clone(), enc_output[:, 1:].clone()

        return word_vecs, seq_vecs
