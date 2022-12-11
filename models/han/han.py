import numpy as np
import torch
from sklearn.utils import shuffle
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
import gc

import models.transformer as tf
from models.han.attention import HANAttention, AddAttention, PeriodicalAttention
from models.transformer import generate_square_subsequent_mask
from utils import gen_aligned_tensor, next_batch


class HAN(nn.Module):
    def __init__(self, d_point, d_traj, attn_size,
                 time_fea_size, vis_fea_size,
                 cell_num, max_seq_len,
                 start_symbol, end_symbol, dev):
        super(HAN, self).__init__()
        self.vis_fea_size = vis_fea_size
        self.time_fea_size = time_fea_size
        self.fea_size = time_fea_size + vis_fea_size
        self.cell_embed = nn.Embedding(cell_num, d_point - time_fea_size + 1)
        self.d_point = d_point
        self.d_traj = d_traj
        self.attn_size = attn_size

        # use rnn instead
        # self.h_size = 64
        # self.traj_encoder = nn.GRU(input_size=self.d_point,
        #                            hidden_size=self.h_size,
        #                            num_layers=2,
        #                            batch_first=False)

        self.point_transformer = tf.Seq2SeqTransformer(num_encoder_layers=3,
                                                       num_decoder_layers=3,
                                                       emb_size=self.d_point,
                                                       nhead=8,
                                                       dim_feedforward=64,
                                                       dropout=0.1).to(dev)

        self.fc1 = nn.Linear(self.d_point * max_seq_len + vis_fea_size, self.d_traj).to(dev)
        self.fc2 = nn.Linear(self.d_traj, self.d_point * max_seq_len).to(dev)
        self.cell_proj = nn.Linear(self.d_point, cell_num).to(dev)

        self.traj_transformer = tf.Seq2SeqTransformer(num_encoder_layers=3,
                                                      num_decoder_layers=3,
                                                      emb_size=self.d_traj,
                                                      nhead=8,
                                                      dim_feedforward=64,
                                                      dropout=0.1).to(dev)
        # self.slice_transformer = tf.Seq2SeqTransformer(num_encoder_layers=3,
        #                                                num_decoder_layers=3,
        #                                                emb_size=self.d_point,
        #                                                nhead=6,
        #                                                dim_feedforward=64,
        #                                                dropout=0.1).to(dev)

        # self.point_attention = HANAttention(d_model=self.d_point,
        #                                     att_size=self.attn_size).to(dev)
        self.traj_attention = PeriodicalAttention(d_traj=self.d_traj, att_size=64).to(dev)

        self.dev = dev

        self.start_symbol = start_symbol
        self.end_symbol = end_symbol

    def forward(self, encoder_inputs, slice_len, PAD_IDX):
        # create the begin and end symbol
        # BOT = torch.zeros(1, self.time_fea_size).fill_(self.start_symbol).type(torch.long).to(self.dev)
        # EOT = torch.zeros(1, self.time_fea_size).fill_(self.end_symbol).type(torch.long).to(self.dev)

        # create the begin and end symbol
        BOT = [self.start_symbol] * self.d_traj

        # get trajectory embedding
        batch_size = 64
        trajectory_vecs = []
        for batch in next_batch(encoder_inputs, batch_size):
            src = torch.transpose(batch, 0, 1)
            src = embedding(self.cell_embed, src, self.dev)

            # use gru
            # input = pack_padded_sequence(input, valid_len, enforce_sorted=False)
            # output, hidden = self.rnn(input)

            # generate mask and train
            src_mask, src_key_padding_mask = create_src_mask(src, self.dev, PAD_IDX)

            # TODO: validate the EOT
            out = self.point_transformer.encode(src, src_mask, src_key_padding_mask)  # (max_valid_len, bs, d_point)
            # TODO:verify reshape correctness
            out = torch.transpose(out, 0, 1).reshape(src.shape[1], -1)  # (bs, max_valid_len * d_point)

            # TODO:concat with vision output
            traj_embed = out

            # change dimension
            traj_embed = self.fc1(traj_embed)
            trajectory_vecs += [x.tolist() for x in traj_embed]  # (d_traj, )
        # trajectory_vecs : (seq_num, d_traj)

        del encoder_inputs, batch, src, src_mask, src_key_padding_mask, out, traj_embed
        gc.collect()

        # restore the trajectories into time slices
        index = 0
        sliced_trajs = []
        for i in slice_len:
            trajs = trajectory_vecs[index:index+i]
            sliced_trajs.append(trajs)
            index+=i

        del trajectory_vecs
        gc.collect()

        # pad time slices
        sliced_trajs, valid_len = gen_aligned_tensor(sliced_trajs, fill_value=PAD_IDX)
        sliced_trajs = torch.from_numpy(sliced_trajs).float().to(self.dev)  # (slice_num, traj_num, d_traj)
        # valid_len = torch.from_numpy(valid_len).long().to(self.dev)  # (slice_num,)
        # seq_len = sliced_trajs.shape[1]
        # sliced_trajs = torch.Tensor(sliced_trajs)  # (slice_num, traj_num, d_traj)

        # get batch first attention
        # traj_embedding, _ = self.point_attention(sliced_trajs)  # (slice_num, traj_num, d_traj)

        # encode slices
        sliced_trajs = torch.transpose(sliced_trajs, 0, 1)
        src_mask, src_key_padding_mask = create_src_mask(sliced_trajs, self.dev, PAD_IDX)
        slice_vecs = self.traj_transformer.encode(sliced_trajs, src_mask,
                                                   src_key_padding_mask)  # (slice_num, traj_num, d_traj)

        del sliced_trajs
        gc.collect()

        # TODO: prediction(including generate label) & pattern learn

        # TODO: attention between slices: 4 hours near and 24 hours before, add attention & other
        slice_vecs = self.traj_attention(slice_vecs)

        # TODO: greedy decode choice, need predict_step
        tgt = slice_vecs.transpose(0,1).tolist()
        gc.collect()
        tgt_input = torch.Tensor([[BOT.copy()]+x[:-1] for x in tgt]).transpose(0,1).to(self.dev)   # shift right
        tgt_mask, tgt_padding_mask = create_tgt_mask(tgt_input, self.dev, PAD_IDX)
        decoded_slices = self.traj_transformer.decode(tgt_input, slice_vecs, tgt_mask=tgt_mask,
                                                      tgt_key_padding_mask=tgt_padding_mask,
                                                      memory_key_padding_mask=src_key_padding_mask)
        # (slice_num, traj_num, d_traj)

        del slice_vecs, src_mask, src_key_padding_mask,tgt,tgt_input,tgt_mask,tgt_padding_mask
        gc.collect()

        decoded_slices = torch.transpose(decoded_slices, 0, 1)

        # flatten time slices
        # traj_embeds = []
        # for traj in decoded_slices:
        #     traj_embeds += traj
        traj_embeds = decoded_slices[-1][:50]  #内存不够，暂时生成前50条轨迹

        del decoded_slices
        gc.collect()

        # generate trajectory using greedy decode
        # -- what is the memory? from the last layer, i.e. the "traj" from decoded_slices
        max_len = 100
        recovered_trajs = []
        for memory in traj_embeds:
            # change dimension
            memory = self.fc2(memory).reshape(-1,1,self.d_point)
            # get the BOT
            BOT = torch.zeros(1, self.time_fea_size).fill_(self.start_symbol).type(torch.long).to(self.dev)
            ys = embedding(self.cell_embed, BOT, self.dev).unsqueeze(0)
            for i in range(max_len - 1):
                memory = memory.to(self.dev)
                tgt_mask = (generate_square_subsequent_mask(ys.size(0), self.dev)
                            .type(torch.bool)).to(self.dev)
                
                out = self.point_transformer.g_decode(ys, memory, tgt_mask)
                word = out.transpose(0, 1)[:,-1:]
                ys = torch.cat([ys,word], dim=0)
                # what is EOT? -- all zero
                if (word.cpu().detach().numpy() == 0).all():
                    break
            recovered_trajs.append(ys)
        recovered_trajs = torch.Tensor(recovered_trajs)

        # TODO: predict time & cell number

        return self.cell_proj(recovered_trajs)


def create_src_mask(src, dev, PAD_IDX):
    src_seq_len = src.shape[0]
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=dev).type(torch.bool)
    temp_src = torch.matmul(torch.abs(src), torch.ones((src.shape[-1], 1), device=dev)).squeeze(-1)
    src_padding_mask = (temp_src == PAD_IDX * src.shape[-1]).transpose(0, 1)
    return src_mask, src_padding_mask


def create_tgt_mask(tgt, dev, PAD_IDX):
    tgt_seq_len = tgt.shape[0]
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, dev)
    temp_tgt = torch.matmul(torch.abs(tgt), torch.ones((tgt.shape[-1], 1), device=dev)).squeeze(-1)
    a = torch.zeros(tgt.shape[1]).type(torch.bool).unsqueeze(0).to(dev)
    tgt_padding_mask = torch.cat((a, (temp_tgt == PAD_IDX * tgt.shape[-1])[1:, :]), dim=0).transpose(0, 1)
    return tgt_mask, tgt_padding_mask

def embedding(embed,input,dev):
    # # 拆分成(batch_size, seq_len, 1) 和 (batch_size, seq_len, frame_dim-1)
    # input = torch.split(input,(1,frame_dim-1),dim=2)

    # embed_input (batch_size, seq_len)
    if len(input.shape) == 3:
        embed_input=input[:,:,0]
    elif len(input.shape) == 2:
        embed_input = input[:, 0]
    # print(embed_input.shape)
    embed_input = torch.LongTensor(embed_input.cpu().numpy())
    embed_output = embed(embed_input).to(dev)

    if len(input.shape) == 3:
        output=torch.cat((embed_output,input[:,:,1:]),dim=2)
    elif len(input.shape) == 2:
        output=torch.cat((embed_output,input[:,1:]),dim=1)

    return output
