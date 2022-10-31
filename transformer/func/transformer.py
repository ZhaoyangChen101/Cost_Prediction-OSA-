import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from func.utils import seq_padding, position_idx, cost_cal
import numpy as np


class TransformerDataset(Dataset):
    def __init__(self, combine, max_len_i=187, max_len_tc=70):
        self.max_len_i = max_len_i
        self.max_len_tc = max_len_tc

        self.input_cost = combine[0]  # input cost
        self.age = combine[1]  # age
        self.gender = combine[2]  # gender
        self.diff = combine[3]  # diff
        self.department = combine[4]  # department
        self.specialist = combine[5]  # specialist
        self.visit_type = combine[6]  # visit_type
        self.input_target_cost = combine[7]  # input target cost
        self.target_cost = combine[8]  # target cost

    def __getitem__(self, index):
        """
        return: input cost, age, gender, diff, department, specialist, visit type, position,
                segment, input target cost, and target cost.
        """
        # cut data
        input_cost = self.input_cost[index]
        age = self.age[index]
        gender = self.gender[index]
        diff = self.diff[index]
        department = self.department[index]
        specialist = self.specialist[index]
        visit_type = self.visit_type[index]
        input_target_cost = self.input_target_cost[index]
        target_cost = self.target_cost[index]

        # extract data
        input_cost = input_cost[(-self.max_len_i + 1):]
        age = age[(-self.max_len_i + 1):]
        gender = gender[(-self.max_len_i + 1):]
        diff = diff[(-self.max_len_i + 1):]
        department = department[(-self.max_len_i + 1):]
        specialist = specialist[(-self.max_len_i + 1):]
        visit_type = visit_type[(-self.max_len_i + 1):]

        # avoid data cut with first element to be 'SEP'
        # 'SEP' -> 52, 'CLS' -> 51
        if input_cost[0] != 52:
            input_cost = np.append(np.array([51]), input_cost)
            age = np.append(np.array(age[0]), age)
            gender = np.append(np.array(gender[0]), gender)
            diff = np.append(np.array(diff[0]), diff)
            department = np.append(np.array(department[0]), department)
            specialist = np.append(np.array(specialist[0]), specialist)
            visit_type = np.append(np.array(visit_type[0]), visit_type)
        else:
            input_cost[0] = 51

        # get position code
        position_i = position_idx(input_cost)
        position_t = position_idx(input_target_cost)

        # pad input data for encoder part of transformer
        input_cost = seq_padding(input_cost, self.max_len_i, symbol=0)
        age = seq_padding(age, self.max_len_i, symbol=0)
        gender = seq_padding(gender, self.max_len_i, symbol=0)
        diff = seq_padding(diff, self.max_len_i, symbol=0)
        department = seq_padding(department, self.max_len_i, symbol=0)
        specialist = seq_padding(specialist, self.max_len_i, symbol=0)
        visit_type = seq_padding(visit_type, self.max_len_i, symbol=0)
        position_i = seq_padding(position_i, self.max_len_i, symbol=0)

        # pad input target cost, position data for input target cost, target cost
        input_target_cost = seq_padding(input_target_cost, self.max_len_tc, symbol=0)
        position_t = seq_padding(position_t, self.max_len_tc, symbol=0)
        target_cost = seq_padding(target_cost, self.max_len_tc, symbol=0)

        return (torch.LongTensor(input_cost), torch.LongTensor(age), torch.LongTensor(gender),
                torch.LongTensor(diff), torch.LongTensor(department), torch.LongTensor(specialist),
                torch.LongTensor(visit_type), torch.LongTensor(position_i)), \
               (torch.LongTensor(input_target_cost), torch.LongTensor(position_t)), torch.LongTensor(target_cost)

    def __len__(self):
        return len(self.input_cost)


class TransformerModel(nn.Module):
    """
    Model from "A detailed guide to Pytorch's nn.Transformer() module.", by
    Daniel Melchor: https://medium.com/p/c80afbc9ffb1/
    """

    # Constructor
    def __init__(
            self,
            cost_vocab_size,
            age_vocab_size,
            gender_vocab_size,
            diff_vocab_size,
            department_vocab_size,
            specialist_vocab_size,
            visit_type_vocab_size,
            max_pos,
            emb_size,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            dropout_p,
    ):
        super().__init__()

        # INFO
        self.model_type = "Transformer"
        self.emb_size = emb_size

        self.dim_model = emb_size  # the emb_size stays the same

        # input embedding
        self.cost_embedding = nn.Embedding(cost_vocab_size, emb_size)
        self.age_embedding = nn.Embedding(age_vocab_size, emb_size)
        self.gender_embedding = nn.Embedding(gender_vocab_size, emb_size)
        self.diff_embedding = nn.Embedding(diff_vocab_size, emb_size)
        self.department_embedding = nn.Embedding(department_vocab_size, emb_size)
        self.specialist_embedding = nn.Embedding(specialist_vocab_size, emb_size)
        self.visit_type_embedding = nn.Embedding(visit_type_vocab_size, emb_size)
        # self.seg_embedding = nn.Embedding(seg_vocab_size, emb_size)
        self.pos_embedding = nn.Embedding(max_pos, emb_size). \
            from_pretrained(embeddings=self._init_posi_embedding(max_pos, emb_size))

        # target embedding
        self.tgt_embedding = nn.Embedding(cost_vocab_size, emb_size)

        self.transformer = nn.Transformer(
            d_model=self.dim_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout_p,
        )

        self.out = nn.Linear(self.dim_model, cost_vocab_size)

    def forward(self, src, target, device, tgt_mask=None, src_pad_mask=None, tgt_pad_mask=None):
        # Src size must be (num_of_variables, batch_size, src sequence length)
        # Tgt size must be (batch_size, tgt sequence length)

        # put data to device, make sure dtype=torch.int64
        # inputs
        cost = src[0].to(device, dtype=torch.int64)
        age = src[1].to(device, dtype=torch.int64)
        gender = src[2].to(device, dtype=torch.int64)
        diff = src[3].to(device, dtype=torch.int64)
        department = src[4].to(device, dtype=torch.int64)
        specialist = src[5].to(device, dtype=torch.int64)
        visit_type = src[6].to(device, dtype=torch.int64)
        pos_i = src[7].to(device, dtype=torch.int64)
        # target
        tgt = target[0].to(device, dtype=torch.int64)
        pos_t = target[1].to(device, dtype=torch.int64)

        batchsize = cost.size(0)
        seqlen = cost.size(1)
        seqlen_t = tgt.size(1)

        # embedding
        # inputs
        cost_emb = self.cost_embedding(cost)
        age_emb = self.age_embedding(age)
        gender_emb = self.gender_embedding(gender)
        diff_emb = self.diff_embedding(diff)
        department_emb = self.department_embedding(department)
        specialist_emb = self.specialist_embedding(specialist)
        visit_type_emb = self.visit_type_embedding(visit_type)
        pos_i_emb = self.pos_embedding(pos_i)
        # target
        tgt_emb = self.tgt_embedding(tgt)
        pos_t_emb = self.pos_embedding(pos_t)

        # ensure correct shape of 7 embeddings:
        # .contiguous.view()
        # inputs
        cost_emb = cost_emb.contiguous().view(batchsize, seqlen, -1)
        age_emb = age_emb.contiguous().view(batchsize, seqlen, -1)
        gender_emb = gender_emb.contiguous().view(batchsize, seqlen, -1)
        diff_emb = diff_emb.contiguous().view(batchsize, seqlen, -1)
        department_emb = department_emb.contiguous().view(batchsize, seqlen, -1)
        specialist_emb = specialist_emb.contiguous().view(batchsize, seqlen, -1)
        visit_type_emb = visit_type_emb.contiguous().view(batchsize, seqlen, -1)
        pos_i_emb = pos_i_emb.contiguous().view(batchsize, seqlen, -1)
        # target
        tgt_emb = tgt_emb.contiguous().view(batchsize, seqlen_t, -1)
        pos_t_emb = pos_t_emb.contiguous().view(batchsize, seqlen_t, -1)

        # summation of embeddings
        src = cost_emb + age_emb + gender_emb + diff_emb + department_emb + \
              specialist_emb + visit_type_emb + pos_i_emb
        tgt = tgt_emb + pos_t_emb

        # permute to obtain size [sequence length, batch_size, dim_model]
        # .contiguous(), which is required by the device mps not by cpu.
        src = src.permute(1, 0, 2).contiguous()
        tgt = tgt.permute(1, 0, 2).contiguous()

        # Transformer blocks - Out size = [sequence length, batch_size, num_tokens]
        transformer_out = self.transformer(src, tgt, tgt_mask=tgt_mask, src_key_padding_mask=src_pad_mask,
                                           tgt_key_padding_mask=tgt_pad_mask)
        # Linear layer
        out = self.out(transformer_out)

        return out

    def _init_posi_embedding(self, max_position_embedding: int, hidden_size: int) -> torch.Tensor:
        # https://github.com/yikuanli/BEHRT/blob/master/task/NextVIsit-12month.ipynb
        def even_code(pos, idx):
            return np.sin(pos / (10000 ** (2 * idx / hidden_size)))

        def odd_code(pos, idx):
            return np.cos(pos / (10000 ** (2 * idx / hidden_size)))

        # initialize position embedding table
        lookup_table = np.zeros((max_position_embedding, hidden_size), dtype=np.float32)

        # reset table parameters with hard encoding
        # set even dimension
        for pos in range(max_position_embedding):
            for idx in np.arange(0, hidden_size, step=2):
                lookup_table[pos, idx] = even_code(pos, idx)
        # set odd dimension
        for pos in range(max_position_embedding):
            for idx in np.arange(1, hidden_size, step=2):
                lookup_table[pos, idx] = odd_code(pos, idx)

        return torch.tensor(lookup_table)

    def get_tgt_mask(self, size: int) -> torch.tensor:
        # https://towardsdatascience.com/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
        # Generates a squeare matrix where the each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size) == 1)  # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf'))  # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0))  # Convert ones to 0
        # EX for size=5:
        # [[0., -inf, -inf, -inf, -inf],
        #  [0.,   0., -inf, -inf, -inf],
        #  [0.,   0.,   0., -inf, -inf],
        #  [0.,   0.,   0.,   0., -inf],
        #  [0.,   0.,   0.,   0.,   0.]]

        return mask

    def create_pad_mask(self, matrix: torch.tensor, pad_token: int) -> torch.Tensor:
        # https://towardsdatascience.com/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
        # If matrix = [1,2,3,0,0,0] where pad_token=0, the result mask is
        # [False, False, False, True, True, True]
        return (matrix == pad_token)


class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()

    def forward(self, output, target, id2cost, cost_tensor, params, device):
        # output: (sequence length, batch_size, num_tokens)
        # permute output into two different dimensions
        out1_p = output.permute(1, 2, 0)  # out1_p (batch_size, num_tokens, seq_len)
        out2 = F.softmax(output, dim=2)
        out2_p = out2.permute(1, 0, 2)  # out2_p (batch_size, seq_len, num_tokens)

        # loss calculation
        loss_1 = F.cross_entropy(out1_p, target)
        annual_true_costs, annual_expected_costs = cost_cal(target, out2_p, id2cost, cost_tensor, device)
        annual_true_costs = annual_true_costs.to(device)
        loss_2 = F.mse_loss(annual_expected_costs, annual_true_costs)
        # three types of scaling methods
        if params['func'] == 'log10':
            loss_2 = torch.log10(loss_2)
        elif params['func'] == 'ln':
            loss_2 = torch.log(loss_2)
        elif params['func'] == 'harmonic_mean':
            total_loss = 2 * loss_1 * loss_2 / (loss_1 + loss_2)
            return total_loss

        total_loss = loss_1 + loss_2

        return total_loss


