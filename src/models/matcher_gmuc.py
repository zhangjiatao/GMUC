import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from .matcher import Matcher, LayerNormalization, SupportEncoder, QueryEncoder


class EmbedMatcher_GMUC(Matcher):
    def __init__(self, arg, num_symbols, embed=None):
        super(EmbedMatcher_GMUC, self).__init__(arg, num_symbols, embed)
        d_model = self.embed_dim * 2

        self.matchnet_mean = MatchNet(d_model, self.process_steps)
        self.matchnet_var = MatchNet(d_model, self.process_steps)

        self.gcn_w = nn.Linear(2*self.embed_dim, self.embed_dim)
        self.gcn_w_var = nn.Linear(2*self.embed_dim, self.embed_dim)

    def scoreOp(self, support, support_meta, query,  query_meta):

        query_left_connections, query_left_degrees, query_right_connections, query_right_degrees = query_meta
        support_left_connections, support_left_degrees, support_right_connections, support_right_degrees = support_meta

        # ======================== 1. Guassin Neighbor Encoder ========================
        query_left, query_var_left = self.neighbor_encoder(query_left_connections, query_left_degrees)
        query_right, query_var_right = self.neighbor_encoder(query_right_connections, query_right_degrees)
        support_left, support_var_left = self.neighbor_encoder(support_left_connections, support_left_degrees)
        support_right, support_var_right = self.neighbor_encoder(support_right_connections, support_right_degrees)
        support_confidence = support[:, 2] # get confidence of support set (suporot_num)
        # ----------------- mean encoder ------------------
        query_neighbor = torch.cat((query_left, query_right), dim=-1)  # tanh
        support_neighbor = torch.cat((support_left, support_right), dim=-1)  # tanh
        support = support_neighbor
        query = query_neighbor
        support_g = self.support_encoder(support) # (3, 200)
        query_g = self.support_encoder(query) # (batch, 2 * embedd_dim)
        
        # position encoder and decoder
        support_g_0 = support_g.view(3, 1, 2 * self.embed_dim) # input of encoder
        support_g_encoder, support_g_state = self.set_rnn_encoder(support_g_0)

        # decoder
        support_g_decoder = support_g_encoder[-1].view(1, -1, 2 * self.embed_dim)
        support_g_decoder_state = support_g_state
        decoder_set = []
        for idx in range(3): 
            support_g_decoder, support_g_decoder_state = self.set_rnn_decoder(support_g_decoder, support_g_decoder_state)
            decoder_set.append(support_g_decoder)
        decoder_set = torch.cat(decoder_set, dim=0) # output of decoder 
        ae_loss = nn.MSELoss()(support_g_0, decoder_set.detach()) # calculate loss of encoder input and decoder output
        # encoder
        support_g_encoder = support_g_encoder.view(3, 2 * self.embed_dim)
        support_g_encoder = support_g_0.view(3, 2 * self.embed_dim) + support_g_encoder # output of encoder

        # add attention
        support_g_att = self.set_att_W(support_g_encoder).tanh()
        att_w = self.set_att_u(support_g_att)
        att_w = self.softmax(att_w)
        support_g_encoder = torch.matmul(support_g_encoder.transpose(0, 1), att_w)
        support_g_encoder = support_g_encoder.transpose(0, 1)
        support_g_encoder = support_g_encoder.view(1, 2 * self.embed_dim)

        support = support_g_encoder
        query = query_g

        # ----------------- variance encoder ------------------
        query_var_neighbor = torch.cat((query_var_left, query_var_right), dim=-1)  # tanh
        support_var_neighbor = torch.cat((support_var_left, support_var_right), dim=-1)  # tanh
        support_var = support_var_neighbor
        query_var = query_var_neighbor
        support_var = torch.mean(support_var_neighbor, dim=0, keepdim=True)

        #  ======================== 2. Guassin Matching Networks ========================
        # ----------------- mean match ------------------
        matching_scores = self.matchnet_mean(support, None, query, None)
        # ----------------- varaince match ------------------
        matching_scores_var = self.matchnet_var(support_var, None, query_var, None)
        matching_scores_var = torch.sigmoid(self.w + matching_scores_var + self.b)

        return matching_scores, matching_scores_var, ae_loss


     
    def forward(self, support, support_meta, query,  query_meta, false, false_meta):

        query_scores, query_scores_var, query_ae_loss = self.scoreOp(support, support_meta, query,  query_meta)
        false_scores, false_scores_var, false_ae_loss = self.scoreOp(support, support_meta, false, false_meta)

        if self.neg_nums != 1:
            false_scores = false_scores.reshape((query_scores.shape[0], self.neg_nums)) # resize
            false_scores = torch.mean(false_scores, dim = 1)
            false_scores = false_scores.reshape((query_scores.shape[0])) # resize

        query_confidence = query[:,2]
        zero_torch = torch.zeros(query_confidence.shape).cuda()
        ones_torch = torch.ones(query_confidence.shape).cuda()
        query_conf_mask = torch.where(query_confidence < 0.5, zero_torch, ones_torch)
        # ------ MSE loss -------
        mae_loss = (query_scores_var - query_confidence)**2
        mae_loss = self.mae_weight * mae_loss.sum()
        # ------ rank loss ------ 
        rank_loss = self.margin - (query_scores - false_scores)
        if self.if_conf:
            rank_loss = torch.mean(F.relu(rank_loss) * query_conf_mask) # rank loss 
        else:
            rank_loss = torch.mean(F.relu(rank_loss))
        rank_loss = self.rank_weight * rank_loss
        # ------ lstem loss ------ 
        ae_loss = self.ae_weight * query_ae_loss # lstm aggregation loss
        # ------ over all loss ------ 
        loss = rank_loss +  mae_loss + ae_loss
        return loss



class MatchNet(nn.Module):
    def __init__(self, input_dim, process_step=4):
        super(MatchNet, self).__init__()
        self.input_dim = input_dim
        self.process_step = process_step
        self.process = nn.LSTMCell(input_dim, 2 * input_dim)


    def forward(self, support_mean, support_var, query_mean, query_var):
        assert support_mean.size()[1] == query_mean.size()[1]

        batch_size = query_mean.size()[0]
        h_r = Variable(torch.zeros(batch_size, 2 * self.input_dim)).cuda()
        c = Variable(torch.zeros(batch_size, 2 * self.input_dim)).cuda()

        for step in range(self.process_step):
            h_r_, c = self.process(query_mean, (h_r, c))
            h = query_mean + h_r_[:, :self.input_dim]  # (batch_size, query_dim)

            attn = F.softmax(torch.matmul(h, support_mean.t()), dim=1)

            r = torch.matmul(attn, support_mean)  # (batch_size, support_dim)
            h_r = torch.cat((h, r), dim=1)

        matching_scores = torch.matmul(h, support_mean.t()).squeeze()
        return matching_scores
