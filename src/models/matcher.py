import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

class Matcher(nn.Module):
    def __init__(self, arg, num_symbols, embed=None):
        super(Matcher, self).__init__()
        for k, v in vars(arg).items(): setattr(self, k, v)
        self.pad_idx = num_symbols
        self.symbol_emb = nn.Embedding(num_symbols + 1, self.embed_dim, padding_idx=num_symbols)
        self.symbol_var_emb = nn.Embedding(num_symbols + 1, self.embed_dim, padding_idx=num_symbols)
        self.num_symbols = num_symbols
        self.layer_norm = LayerNormalization(2 * self.embed_dim)
        dropout = self.dropout

        if self.random_embed:
            self.use_pretrain = False
        else:
            self.use_pretrain = True

        # self.gnn_w = nn.Linear(2 * self.embed_dim, self.embed_dim)
        # self.gnn_b = nn.Parameter(torch.FloatTensor(self.embed_dim))

        self.dropout = nn.Dropout(dropout)
        
        # aggregation encoder and decoder
        self.set_rnn_encoder = nn.LSTM(2 * self.embed_dim, 2 * self.embed_dim, 1, bidirectional = False)
        self.set_rnn_decoder = nn.LSTM(2 * self.embed_dim, 2 * self.embed_dim, 1, bidirectional = False)

        self.set_FC_encoder = nn.Linear(3 * 2 * self.embed_dim, 2 * self.embed_dim)
        self.set_FC_decoder = nn.Linear(2 * self.embed_dim, 3 * 2 * self.embed_dim)

        # neighbor encoder attention
        self.neigh_att_W = nn.Linear(2 * self.embed_dim, self.embed_dim)
        self.neigh_att_u = nn.Linear(self.embed_dim, 1)
        self.neigh_var_att_W = nn.Linear(2 * self.embed_dim, self.embed_dim)
        self.neigh_var_att_u = nn.Linear(self.embed_dim, 1)
        # aggregation attention
        self.set_att_W = nn.Linear(2 * self.embed_dim, self.embed_dim)
        self.set_att_u = nn.Linear(self.embed_dim, 1)
        self.set_var_att_W = nn.Linear(2 * self.embed_dim, self.embed_dim)
        self.set_var_att_u = nn.Linear(self.embed_dim, 1)

        self.bn = nn.BatchNorm1d(2 * self.embed_dim)
        self.softmax = nn.Softmax(dim=1)

        # self.support_g_W = nn.Linear(4 * self.embed_dim, 2 * self.embed_dim)

        self.FC_query_g = nn.Linear(2 * self.embed_dim, 2 * self.embed_dim)
        self.FC_support_g_encoder = nn.Linear(2 * self.embed_dim, 2 * self.embed_dim)

        # init.xavier_uniform(self.symbol_emb.weight.data)
        # init.xavier_normal_(self.gnn_w.weight)
        init.xavier_normal_(self.neigh_att_W.weight)
        init.xavier_normal_(self.neigh_att_u.weight)
        init.xavier_normal_(self.set_att_W.weight)
        init.xavier_normal_(self.set_att_u.weight)
        # init.xavier_normal_(self.support_g_W.weight)
        # init.constant_(self.gnn_b, 0)

        init.xavier_normal_(self.FC_query_g.weight)
        init.xavier_normal_(self.FC_support_g_encoder.weight)

        # Here is symbol to embedding
        if self.use_pretrain:
            self.symbol_emb.weight.data.copy_(torch.from_numpy(embed))
            self.symbol_var_emb.weight.data.copy_(torch.from_numpy(embed))
            if not finetune:
                self.symbol_emb.weight.requires_grad = False
                self.symbol_var_emb.weight.requires_grad = False

        d_model = self.embed_dim * 2
        self.support_encoder = SupportEncoder(d_model, 2 * d_model, dropout)
        self.query_encoder = QueryEncoder(d_model, self.process_steps)

        # my code
        self.w = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.b = nn.Parameter(torch.FloatTensor(1), requires_grad=True)

    def neighbor_encoder(self, connections, num_neighbors):

        num_neighbors = num_neighbors.unsqueeze(1)
        relations = connections[:, :, 0].squeeze(-1).long() # neighbor relation
        entities = connections[:, :, 1].squeeze(-1).long() # neighor entity
        confidences = connections[:, :, 2]# 获取neighor confidence (batch, neighbor)

        (batch_size, neighbor_size) = confidences.shape
        confidences = confidences.reshape((batch_size, neighbor_size, 1)).repeat((1, 1, 2 * self.embed_dim))

        rel_embeds = self.dropout(self.symbol_emb(relations))  # (batch, neighbor_num, embed_dim)
        ent_embeds = self.dropout(self.symbol_emb(entities))  # (batch, neighbor_num, embed_dim)
        concat_embeds = torch.cat((rel_embeds, ent_embeds), dim=-1)  # (batch, neighbor_num, 2*embed_dim) [1001, 30, 200]

        rel_var_embeds = self.dropout(self.symbol_var_emb(relations))  # (batch, neighbor_num, embed_dim)
        ent_var_embeds = self.dropout(self.symbol_var_emb(entities))  # (batch, neighbor_num, embed_dim)
        concat_var_embeds = torch.cat((rel_var_embeds, ent_var_embeds), dim=-1)  # (batch, neighbor_num, 2*embed_dim) [1001, 30, 200]

        # # consider triple quantlity
        # concat_embeds = concat_embeds * confidences
        # concat_var_embeds = concat_embeds * confidences

        # position embedding
        out = self.neigh_att_W(concat_embeds).tanh()
        att_w = self.neigh_att_u(out)
        att_w = self.softmax(att_w).view(concat_embeds.size()[0], 1, 30)
        out = torch.bmm(att_w, ent_embeds).view(concat_embeds.size()[0], self.embed_dim)
        # print('[INFO] neighbor_encoder: output size', out.shape)  # (batch, embed_dim) [1001, 100]

        # variance embedding
        out_var = self.neigh_var_att_W(concat_var_embeds).tanh()
        att_var_w = self.neigh_var_att_u(out_var)
        att_var_w = self.softmax(att_var_w).view(concat_var_embeds.size()[0], 1, 30)
        out_var = torch.bmm(att_var_w, ent_embeds).view(concat_var_embeds.size()[0], self.embed_dim)

        return out.tanh(), out_var.tanh()


    def scoreOp(self, support, support_meta, query,  query_meta):
        raise NotImplementedError


    def forward(self, support, support_meta, query,  query_meta, false, false_meta):
        raise NotImplementedError



class LayerNormalization(nn.Module):
    ''' Layer normalization module '''

    def __init__(self, d_hid, eps=1e-3):
        super(LayerNormalization, self).__init__()

        self.eps = eps
        self.a_2 = nn.Parameter(torch.ones(d_hid), requires_grad=True)
        self.b_2 = nn.Parameter(torch.zeros(d_hid), requires_grad=True)

    def forward(self, z):
        if z.size(1) == 1:
            return z

        mu = torch.mean(z, keepdim=True, dim=-1)
        sigma = torch.std(z, keepdim=True, dim=-1)
        ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
        ln_out = ln_out * self.a_2.expand_as(ln_out) + self.b_2.expand_as(ln_out)

        return ln_out


class SupportEncoder(nn.Module):
    """docstring for SupportEncoder"""

    def __init__(self, d_model, d_inner, dropout=0.1):
        super(SupportEncoder, self).__init__()
        self.proj1 = nn.Linear(d_model, d_inner)
        self.proj2 = nn.Linear(d_inner, d_model)
        self.layer_norm = LayerNormalization(d_model)

        init.xavier_normal_(self.proj1.weight)
        init.xavier_normal_(self.proj2.weight)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        output = self.relu(self.proj1(x))
        output = self.dropout(self.proj2(output))
        return self.layer_norm(output + residual)


class QueryEncoder(nn.Module):
    def __init__(self, input_dim, process_step=4):
        super(QueryEncoder, self).__init__()
        self.input_dim = input_dim
        self.process_step = process_step
        self.process = nn.LSTMCell(input_dim, 2 * input_dim)

    def forward(self, support, query):
        '''
        support: (few, support_dim)
        query: (batch_size, query_dim)
        support_dim = query_dim
        return:
        (batch_size, query_dim)
        '''
        
        assert support.size()[1] == query.size()[1]

        if self.process_step == 0:
            return query

        batch_size = query.size()[0]
        h_r = Variable(torch.zeros(batch_size, 2 * self.input_dim)).cuda()
        c = Variable(torch.zeros(batch_size, 2 * self.input_dim)).cuda()

        for step in range(self.process_step):
            h_r_, c = self.process(query, (h_r, c))
            h = query + h_r_[:, :self.input_dim]  # (batch_size, query_dim)
            attn = F.softmax(torch.matmul(h, support.t()), dim=1)
            r = torch.matmul(attn, support)  # (batch_size, support_dim)
            h_r = torch.cat((h, r), dim=1)

        return h
