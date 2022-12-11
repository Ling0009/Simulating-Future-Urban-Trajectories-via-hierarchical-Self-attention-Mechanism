import os
import pickle
import shutil
import time

import torch
from sklearn.utils import shuffle
from torch import nn
from torch.utils.tensorboard import SummaryWriter

import models.transformer as tf
from models.han.han import HAN,embedding
from utils import next_batch, gen_aligned_tensor

# parameters
num_loc = 9120
train_proportion = 0.8
val_proportion = 0.0

def traj_trans_pretrain(input_datas, embed_layer, model, dev):
    # Parameters
    training_epochs = 200
    early_stopping_round = 10

    # Network Parameters
    # 2 different sequences total
    batch_size = 40
    # the maximum steps for both sequences
    max_n_steps = input_datas.shape   # 942(100m, keep all): out of memory;
                                                        # 485(100m, keep 1/3): OK, Epoch Time: 80s
                                                        # 341(100m, keep 1/5): OK, Epoch Time: 42s
    # each element/frame of the sequence has dimension of 13
    frame_dim = 13
    PAD_IDX = 0

    # build model
    optimizer = torch.optim.Adam(list(model.parameters())+list(embed_layer.parameters()),
                                 lr=0.001, betas=(0.9, 0.98), eps=1e-9)

    trajectoryVecs = []
    min_loss, worse_round = 1e8, 0

    # 删除原有log
    log_path = ["./logs/train_log"]
    for path in log_path:
        if os.path.exists(path):
            shutil.rmtree(path)

    # 初始化tensorbord writer
    writer = SummaryWriter("./logs/train_log")
    for epoch in range(0, training_epochs):  # csy modified 21/9/4 采用utils中函数
        total_start_time = time.time()
        train_losses, val_losses = 0, 0

        for batch in next_batch(shuffle(input_datas), batch_size):
            batch = embedding(embed_layer,batch,dev)
            embedding_, train_loss, val_loss = tf.train(model, optimizer, batch, PAD_IDX, dev)

            train_losses += train_loss
            val_losses += val_loss

        total_end_time = time.time()
        # train_losses /= len(encoder_inputs)
        # val_losses /= len(encoder_inputs)

        trajectoryVecs.append(embedding_)

        print(f"Epoch: {epoch+1}, Train loss: {train_losses:.3f}, Val loss: {val_losses:.3f}, "
              F'Epoch Time: {(total_end_time-total_start_time):.3f}s')

        # 使用tensorbord
        # writer.add_scalars("loss", {'train_loss':train_losses, 'val_loss':val_losses}, epoch)
        # param_dict = {}
        # idx = 0
        # for layer in model.transformer.encoder.layers:
        #     param_dict[f'layer{idx}_linear1_weight'] = layer.linear1.weight
        #     param_dict[f'layer{idx}_linear1_bias'] = layer.linear1.bias
        #     param_dict[f'layer{idx}_linear2_weight'] = layer.linear2.weight
        #     param_dict[f'layer{idx}_linear2_bias'] = layer.linear2.bias
        #     idx += 1
        #
        #
        # for key, value in param_dict.items():
        #     writer.add_histogram(key, value, epoch)

        if early_stopping_round > 0 and val_losses < min_loss:
            min_loss = val_losses
            worse_round = 0
        else:
            worse_round += 1

        if 0 < early_stopping_round <= worse_round:
            print('Early stopping @ epoch %d' % (epoch - worse_round), flush=True)
            break


def train_model():
    # read data
    sliced_data = pickle.load(open('./data/normal_feas', 'rb'))

    # split datasets
    dataset_len = len(sliced_data)
    train_seq = sliced_data[:int(dataset_len * train_proportion)][:11]
    # val_seq = sliced_data[int(dataset_len * train_proportion):int(dataset_len * (train_proportion + val_proportion))]
    # test_seq = sliced_data[int(dataset_len * (train_proportion + val_proportion)):]

    sliced_data = []  #释放内存

    # flatten time slices
    traj_data = []
    slice_len = []
    for traj_slice in train_seq:
        traj_data += traj_slice
        slice_len.append(len(traj_slice))

    start_symbol = 1
    end_symbol = 0
    time_fea_size = 13
    dev = "cpu"
    # dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    PAD_IDX = 0

    # create the begin and end symbol
    EOT = [end_symbol] * time_fea_size
    traj_data = [x+[EOT.copy()] for x in traj_data]

    # pad
    encoder_inputs, valid_len = gen_aligned_tensor(traj_data, fill_value=PAD_IDX)
    encoder_inputs = torch.from_numpy(encoder_inputs).float().to(dev)  # (seq_num, max_valid_len, d_point)
    # valid_len = torch.from_numpy(valid_len).long().to(self.dev)  # (seq_num,)
    seq_len = encoder_inputs.shape[1]

    model = HAN(d_point=32, d_traj=128, attn_size=64,
                time_fea_size=time_fea_size,
                vis_fea_size=0,
                cell_num=num_loc,
                max_seq_len=seq_len,
                start_symbol=start_symbol,
                end_symbol=end_symbol,
                dev=dev)

    # pretrain-stage 1
    # print('Start pretrain stage 1.')
    # traj_trans_pretrain(encoder_inputs, model.cell_embed, model.point_transformer, dev)
    # print('Stop pretrain stage 1.')

    # pretrain-stage 2
    print('Start pretrain stage 2.')
    training_epochs = 10
    early_stopping_round = 5
    min_loss, worse_round = 1e8, 0
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    for epoch in range(training_epochs):
        total_start_time = time.time()

        cell_true = encoder_inputs[:,:,0]

        model.train()
        out = model(encoder_inputs, slice_len, PAD_IDX)
        optimizer.zero_grad()
        criterion = nn.CrossEntropyLoss()
        loss = criterion(cell_true, out)
        train_loss = loss.item()
        loss.backward()
        optimizer.step()

        out = model(encoder_inputs, slice_len, PAD_IDX)
        loss = criterion(cell_true, out)
        val_loss = loss.item()

        total_end_time = time.time()

        print(f"Epoch: {epoch + 1}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}"
              F'Epoch Time: {(total_end_time - total_start_time):.3f}s')

        if early_stopping_round > 0 and val_loss < min_loss:
            min_loss = val_loss
            worse_round = 0
        else:
            worse_round += 1

        if 0 < early_stopping_round <= worse_round:
            print('Early stopping @ epoch %d' % (epoch - worse_round), flush=True)
            break

    print('Stop pretrain stage 2.')


if __name__ == "__main__":
    # traj_trans_pretrain()
    # slice_trans_pretrain()
    train_model()