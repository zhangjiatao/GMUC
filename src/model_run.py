import numpy as np
import random
import torch
from .args import read_args
from .data_process import *
import json
from .data_generator import *
from .models.matcher_gmuc import *
from .input_analysis import *
import torch.nn.functional as F
from collections import defaultdict
from collections import deque
from torch import optim
from torch.autograd import Variable
from tqdm import tqdm
import os
import sys
from visdom import Visdom
import time
import nni

torch.set_num_threads(1)
os.environ['CUDA_VISIBLE_DEVICES']='0'

class Model_Run(object):
    def __init__(self, arg):
        super(Model_Run, self).__init__()

        # log files
        metric_log = './Experiments/' + arg.experiment_name + '/metric.log'
        self.metric_log = open(metric_log, 'w')
        loss_log = './Experiments/' + arg.experiment_name + '/loss.log'
        self.loss_log = open(loss_log, 'w')
        param_log = './Experiments/' + arg.experiment_name + '/param.log'
        self.param_log = open(param_log, 'w')    
        result_log = './Experiments/' + arg.experiment_name + '/result.log'
        self.result_log = open(result_log, 'w')   
            
        for k, v in vars(arg).items(): setattr(self, k, v)
        param = ''
        param += "============= arguments/parameters ============\n"
        for k, v in vars(arg).items():
            param += k + ': ' + str(v) + '\n'
        param += "===================================================\n"
        print(param)
        print(param, file = self.param_log, flush = True)

        # setup random seeds
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.cuda.manual_seed_all(self.random_seed)

        # create needed files
        # self.data_analysis()

        # if load pre-train model
        if self.test or self.random_embed:
            self.load_symbol2id()
        else:
            self.load_embed()
            
        self.num_symbols = len(self.symbol2id.keys()) - 1 # one for 'PAD'
        self.pad_id = self.num_symbols

        # select model
        self.matcher = EmbedMatcher_GMUC(arg, self.num_symbols, embed = self.symbol2vec)
        if self.if_GPU:
            self.matcher.cuda()

        # some parameter
        self.batch_nums = 0 
        self.parameters = filter(lambda p: p.requires_grad, self.matcher.parameters())
        self.optim = optim.Adam(self.parameters, lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[10000], gamma=0.25)

        # load files
        self.ent2id = json.load(open(self.datapath + '/ent2ids'))
        self.num_ents = len(self.ent2id.keys())
        degrees = self.build_graph(max_=self.max_neighbor) 
        self.rel2candidates = json.load(open(self.datapath + '/rel2candidates_all.json')) 
        # load answer dict
        self.e1rel_e2 = defaultdict(list)
        self.e1rel_e2 = json.load(open(self.datapath + '/e1rel_e2.json'))
        # visdom
        self.vis = Visdom(env = self.experiment_name)
        data_name = self.datapath.replace('./data/', '')
        t = self.set_aggregator + ' ' + data_name +' Loss'
        self.win_loss = self.vis.line(X = np.array([0]), 
                                      Y = np.array([0]), 
                                      opts = dict(title = t, xlabel = 'epoch', ylabel = 'loss', legend = ['loss']))
        t = self.set_aggregator+ ' ' + data_name + ' MAE'
        self.win_mae = self.vis.line(X = np.column_stack([np.array([0])] * 4), 
                                     Y = np.column_stack([np.array([0])] * 4), 
                                     opts = dict(title = t, legend = ['MAE', 'gt_mean', 'pred_mean', 'neg_pred_mean'], xlabel = 'epoch'))
        t = self.set_aggregator+ ' ' + data_name + ' HIT'
        self.win_hit = self.vis.line(X = np.column_stack([np.array([0])] * 4), 
                                     Y = np.column_stack([np.array([0])] * 4), 
                                     opts = dict(title = t, legend = ['r_hit_1', 'r_hit_10', 'f_hit_1', 'f_hit_10'], xlabel = 'epoch', ylabel = 'hit'))
        t = self.set_aggregator+ ' ' + data_name + ' MR'
        self.win_mr = self.vis.line(X = np.column_stack([np.array([0])] * 2), 
                                     Y = np.column_stack([np.array([0])] * 2), 
                                     opts = dict(title = t, legend = ['r_mr', 'f_mr'], xlabel = 'epoch', ylabel = 'mr'))

    def load_symbol2id(self): 
        '''
        create relation and entity 2 id
        '''     
        symbol_id = {}
        rel2id = json.load(open(self.datapath + '/relation2ids'))
        ent2id = json.load(open(self.datapath + '/ent2ids'))
        i = 0
        for key in rel2id.keys():
            if key not in ['','OOV']:
                symbol_id[key] = i
                i += 1

        for key in ent2id.keys():
            if key not in ['', 'OOV']:
                symbol_id[key] = i
                i += 1

        symbol_id['PAD'] = i
        self.symbol2id = symbol_id
        self.symbol2vec = None


    def load_embed(self):
        '''
        load pretrain Embedding
        '''
        symbol_id = {}
        rel2id = json.load(open(self.datapath + '/relation2ids'))
        ent2id = json.load(open(self.datapath + '/ent2ids'))

        print ("loading pre-trained embedding...")
        if self.embed_model in ['DistMult', 'TransE', 'ComplEx', 'RESCAL']:
            ent_embed = np.loadtxt(self.datapath + '/embed/entity2vec.' + self.embed_model)
            rel_embed = np.loadtxt(self.datapath + '/embed/relation2vec.' + self.embed_model)

            if self.embed_model == 'ComplEx':
                # normalize the complex embeddings
                ent_mean = np.mean(ent_embed, axis=1, keepdims=True)
                ent_std = np.std(ent_embed, axis=1, keepdims=True)
                rel_mean = np.mean(rel_embed, axis=1, keepdims=True)
                rel_std = np.std(rel_embed, axis=1, keepdims=True)
                eps = 1e-3
                ent_embed = (ent_embed - ent_mean) / (ent_std + eps)
                rel_embed = (rel_embed - rel_mean) / (rel_std + eps)

            assert ent_embed.shape[0] == len(ent2id.keys())
            assert rel_embed.shape[0] == len(rel2id.keys())

            i = 0
            embeddings = []
            for key in rel2id.keys():
                if key not in ['','OOV']:
                    symbol_id[key] = i
                    i += 1
                    embeddings.append(list(rel_embed[rel2id[key],:]))

            for key in ent2id.keys():
                if key not in ['', 'OOV']:
                    symbol_id[key] = i
                    i += 1
                    embeddings.append(list(ent_embed[ent2id[key],:]))

            symbol_id['PAD'] = i
            embeddings.append(list(np.zeros((rel_embed.shape[1],))))
            embeddings = np.array(embeddings)
            assert embeddings.shape[0] == len(symbol_id.keys())

            self.symbol2id = symbol_id
            self.symbol2vec = embeddings


    def build_graph(self, max_=50):
        '''
        load path_graph
        '''
        # self.connections = (np.ones((self.num_ents, max_, 3)) * self.pad_id).astype(int)
        self.connections = (np.ones((self.num_ents, max_, 3)) * self.pad_id)
        self.e1_rele2 = defaultdict(list)
        self.e1_degrees = defaultdict(int)

        with open(self.datapath + '/path_graph') as f:
            lines = f.readlines()
            for line in tqdm(lines):
                e1,rel,e2, s = line.rstrip().split()
                self.e1_rele2[e1].append((self.symbol2id[rel], self.symbol2id[e2], float(s)))
                self.e1_rele2[e2].append((self.symbol2id[rel+'_inv'], self.symbol2id[e1], float(s)))

        degrees = {}
        for ent, id_ in self.ent2id.items():
            neighbors = self.e1_rele2[ent]
            if len(neighbors) > max_:
                neighbors = neighbors[:max_]
            degrees[ent] = len(neighbors)
            self.e1_degrees[id_] = len(neighbors) # add one for self conn
            for idx, _ in enumerate(neighbors):
                self.connections[id_, idx, 0] = _[0] # relation
                self.connections[id_, idx, 1] = _[1] # tail entity
                self.connections[id_, idx, 2] = _[2] # confidence
                # print(self.connections[id_, idx, 2])
        return degrees


    def data_analysis(self):
        rel_triples_dis(self.datapath)
        build_vocab(self.datapath)
        candidate_triples(self.datapath)
        for_filtering(self.datapath, save = True)
        print("data analysis finish")


    def get_meta(self, left, right):
        '''
        get meta data
        '''
        if self.if_GPU:
            left_connections = Variable(torch.Tensor(np.stack([self.connections[_,:,:] for _ in left], axis=0))).cuda()
            left_degrees = Variable(torch.FloatTensor([self.e1_degrees[_] for _ in left])).cuda()
            right_connections = Variable(torch.Tensor(np.stack([self.connections[_,:,:] for _ in right], axis=0))).cuda()
            right_degrees = Variable(torch.FloatTensor([self.e1_degrees[_] for _ in right])).cuda()
        else:
            left_connections = Variable(torch.LongTensor(np.stack([self.connections[_,:,:] for _ in left], axis=0)))
            left_degrees = Variable(torch.FloatTensor([self.e1_degrees[_] for _ in left]))
            right_connections = Variable(torch.LongTensor(np.stack([self.connections[_,:,:] for _ in right], axis=0)))
            right_degrees = Variable(torch.FloatTensor([self.e1_degrees[_] for _ in right]))

        return (left_connections, left_degrees, right_connections, right_degrees)


    def train(self):
        '''
        train model
        '''
        print ('start training...')

        # # load pre-trained model
        # self.matcher.load_state_dict(torch.load('./checkpoints/TEST40000.ckpt'))
        # self.batch_nums = 40001
        # print('[INFO] load finish')

        for data in train_generate(self.datapath, self.neg_nums, self.batch_size, self.few, self.symbol2id, self.ent2id, self.e1rel_e2, self.type_constrain):

            support, query, false, support_left, support_right, query_left, query_right, query_confidence, false_left, false_right = data	
            support_meta = self.get_meta(support_left, support_right)
            query_meta = self.get_meta(query_left, query_right)
            false_meta = self.get_meta(false_left, false_right)

            if self.if_GPU:
                support = Variable(torch.Tensor(support)).cuda()
                query = Variable(torch.Tensor(query)).cuda()
                false = Variable(torch.Tensor(false)).cuda()
                query_confidence = Variable(torch.FloatTensor(query_confidence)).cuda()
            else:
                support = Variable(torch.Tensor(support))
                query = Variable(torch.LongTensor(query))
                false = Variable(torch.LongTensor(false))

            loss = self.matcher(support, support_meta, query, query_meta, false, false_meta)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()    

            if self.batch_nums % 100 == 0:
            # if self.batch_nums % 5 == 0:
                print("Epoch %d | loss: %f " % (self.batch_nums, loss.item()))
                print("Epoch %d | loss: %f " % (self.batch_nums, loss.item()), file = self.loss_log, flush = True)
            if self.batch_nums % self.eval_every == 0:
                print("Epoch %d has finished, saving..." % (self.batch_nums))
                path = './Experiments/' + self.experiment_name + '/checkpoints/' + self.set_aggregator + str(self.batch_nums) + ".ckpt"
                torch.save(self.matcher.state_dict(), path)
            # if self.batch_nums % self.eval_every == 0:
            if self.batch_nums % self.eval_every == 0 and self.batch_nums != 0:
                mean_confidence, mean_var, mae, mse, hits10, hits5, hits1, mrr = self.eval()

            # visdom
            x = np.array([self.batch_nums])
            y = np.array([loss.item()])
            self.vis.line(y, x, self.win_loss, update="append")

            self.batch_nums += 1
            self.scheduler.step()

            if self.batch_nums > self.max_batches:
                break


    def eval(self, mode='test'):
        '''
        test model
        '''
        pre_time = 0
        tot_time_s = time.time()
        self.matcher.eval()

        # load file
        symbol2id = self.symbol2id
        few = self.few
        if mode == 'dev':
            test_tasks = json.load(open(self.datapath + '/dev_tasks.json'))
            test_with_neg_tasks = json.load(open(self.datapath + '/dev_neg_tasks.json')) # include pos and neg examples
        else:
            test_tasks = json.load(open(self.datapath + '/test_tasks.json'))
            test_with_neg_tasks = json.load(open(self.datapath + '/test_neg_tasks.json')) # include pos and neg examples
        rel2candidates = self.rel2candidates
        # task_embed_f = open(self.datapath + "task_embed.txt", "w")

        # res container
        mae, mse, mean_var, mean_neg_var, mean_confidence = [], [], [], [], []
        r_mr, r_mrr, r_hits1, r_hits5, r_hits10 = [], [], [], [], []
        f_mr, f_mrr, f_hits1, f_hits5, f_hits10 = [], [], [], [], []
        support_size = 0
        process_cnt = 0
        num_query = len(test_tasks.keys())
        
        # every test relation task
        for query_ in test_tasks.keys():
            process_cnt += 1
            # if(process_cnt >= 5):
            #     continue
            sys.stdout.write("process : %.3lf \r" % (process_cnt / num_query))
            sys.stdout.flush()

            candidates = rel2candidates[query_] # task rel candidate entity
            # support_triples = test_tasks[query_][:few] # pos suppport
            support_triples = test_with_neg_tasks[query_][:few] # pos and neg support
            
            support_pairs = [[symbol2id[triple[0]], symbol2id[triple[2]], float(triple[3])] for triple in support_triples]
            support_left = [self.ent2id[triple[0]] for triple in support_triples]
            support_right = [self.ent2id[triple[2]] for triple in support_triples]
            support_meta = self.get_meta(support_left, support_right)
            
            if self.if_GPU:
                support = Variable(torch.Tensor(support_pairs)).cuda()
            else:
                support = Variable(torch.Tensor(support_pairs))

            neg_var = 0
            for triple in test_tasks[query_][few:]:
                if triple in support_triples:
                    continue

                true = triple[2] # true tail entity

                # create query 
                query_pairs_all = []
                query_left_all = []
                query_right_all = []
                scores_all = np.array([])
                # add pos
                query_pairs_all.append([symbol2id[triple[0]], symbol2id[triple[2]]])
                query_left_all.append(self.ent2id[triple[0]])
                query_right_all.append(self.ent2id[triple[2]])
                # add neg
                if not self.type_constrain:
                    candidates = list(self.ent2id.keys())  
                for ent in candidates:
                    # if (ent not in self.e1rel_e2[triple[0] + triple[1]]) and ent != true:
                    query_pairs_all.append([symbol2id[triple[0]], symbol2id[ent]])
                    query_left_all.append(self.ent2id[triple[0]])
                    query_right_all.append(self.ent2id[ent])

                # divide val data
                val_batch_num = 1
                val_num = len(candidates) + 1 
                if not self.type_constrain:
                    val_batch_num = 5
                batch_size = int(val_num / val_batch_num)

                for batch_index in range(val_batch_num):
                    
                    # [low_bound, up_bound]
                    low_bound = batch_index * batch_size
                    up_bound = min((batch_index + 1) * batch_size, val_num - 1)

                    query_pairs = query_pairs_all[low_bound: (up_bound + 1)]
                    query_left = query_left_all[low_bound: (up_bound + 1)]
                    query_right = query_right_all[low_bound: (up_bound + 1)]

                    if self.if_GPU:
                        query = Variable(torch.LongTensor(query_pairs)).cuda()
                    else:
                        query = Variable(torch.LongTensor(query_pairs))

                    query_meta = self.get_meta(query_left, query_right)
                    scores, scores_var, _ = self.matcher.scoreOp(support, support_meta, query,  query_meta)

                    # # rank setting
                    # scores = scores_var 
                    # scores = 1.0 - torch.abs(scores_var - float(triple[3]))
                    # scores = scores * scores_var
                    # scores = scores.tanh() + scores_var
                    # if batch_index == 0:
                    #     scores_var[0] = float(triple[3])
                    # scores = scores_var
                    # scores = 1.0 - torch.abs(scores_var - float(triple[3]))
                    if 'N3' in self.datapath:
                        # print('[INFO] N3')
                        scores = self.rank_weight * scores.tanh() + scores_var

                    # score
                    scores.detach()
                    scores = scores.data
                    scores = scores.cpu().numpy()
                    scores_all = np.concatenate((scores_all, scores), axis = 0)
                    # score var
                    scores_var.detach()
                    scores_var = scores_var.data
                    scores_var = scores_var.cpu().numpy()

                    neg_var += float(np.mean(scores_var[1:]))
                    # print(np.mean(scores_var[1:]))
                    # print(neg_var)

                    if batch_index == 0:
                        score_var_pred = float(scores_var[0]) # get pos score
                    
                # mae and mse
                query_confidence = float(triple[3])

                mean_confidence.append(query_confidence)
                mean_var.append(score_var_pred)
                mean_neg_var.append(neg_var / len(test_tasks[query_][few:]))
                tmp_mae = abs(query_confidence - score_var_pred)
                tmp_mse = (query_confidence - score_var_pred) * (query_confidence - score_var_pred)
                mae.append(tmp_mae)
                mse.append(tmp_mse)

                # filter
                f_scores_all = np.array([])
                f_scores_all = np.append(f_scores_all, scores_all[0])# add true score

                for index, ent in enumerate(candidates):
                    if (ent not in self.e1rel_e2[triple[0] + triple[1]]) and ent != true: # filter
                        f_scores_all = np.append(f_scores_all, scores_all[index + 1]) 

                # raw
                r_sort = list(np.argsort(scores_all))[::-1]
                # r_sort = list(np.argsort(scores_all))
                r_rank = r_sort.index(0) + 1
                if r_rank <= 10:
                    r_hits10.append(1.0)
                else:
                    r_hits10.append(0.0)
                if r_rank <= 5:
                    r_hits5.append(1.0)
                else:
                    r_hits5.append(0.0)
                if r_rank <= 1:
                    r_hits1.append(1.0)
                else:
                    r_hits1.append(0.0)
                r_mrr.append(1.0 / r_rank)
                r_mr.append(r_rank)
                # filter
                f_sort = list(np.argsort(f_scores_all))[::-1]
                # f_sort = list(np.argsort(f_scores_all))
                f_rank = f_sort.index(0) + 1
                if f_rank <= 10:
                    f_hits10.append(1.0)
                else:
                    f_hits10.append(0.0)
                if f_rank <= 5:
                    f_hits5.append(1.0)
                else:
                    f_hits5.append(0.0)
                if f_rank <= 1:
                    f_hits1.append(1.0)
                else:
                    f_hits1.append(0.0)
                f_mrr.append(1.0 / f_rank)
                f_mr.append(f_rank)

                tmp_str = '%s\t%s\t%s\t%.3lf\t%d\t%d\t%.3lf\t%.3lf' % (triple[0], triple[1], triple[2], float(triple[3]), r_rank, f_rank, tmp_mae, tmp_mse)
                print(tmp_str, file = self.result_log, flush = True)

        # show result
        res = ''
        res += ('============= Epoch: %d valid =============\n' % self.batch_nums)
        res += ('mean support size: %d\n' % int(support_size / len(test_tasks.keys())))
        res += ('mean confidence: %.3lf, mean variance: %.3lf, neg mean variance: %.3lf\n' % (np.mean(mean_confidence), np.mean(mean_var), np.mean(mean_neg_var)))
        res += ('MAE, MSE: %.3lf, %.3lf\n' % (np.mean(mae), np.mean(mse)))
        res += ('------ raw ------\n')		
        res += ('r_mrr, r_mr, r_hits1, r_fits10: %.3lf, %.3lf, %.3lf, %.3lf\n' % (np.mean(r_mrr), np.mean(r_mr), np.mean(r_hits1), np.mean(r_hits10)))
        res += ('r_mr: {:.3f}\n'.format(np.mean(r_mr)))
        res += ('r_mrr: {:.3f}\n'.format(np.mean(r_mrr)))
        res += ('r_hits1: {:.3f}\n'.format(np.mean(r_hits1)))
        res += ('r_hits5: {:.3f}\n'.format(np.mean(r_hits5)))
        res += ('r_hits10: {:.3f}\n'.format(np.mean(r_hits10)))
        res += ('------ filter ------\n')
        res += ('f_mrr, f_mr, f_hits1, f_fits10: %.3lf, %.3lf, %.3lf, %.3lf\n' % (np.mean(f_mrr), np.mean(f_mr), np.mean(f_hits1), np.mean(f_hits10)))
        res += ('f_mr: {:.3f}\n'.format(np.mean(f_mr)))
        res += ('f_mrr: {:.3f}\n'.format(np.mean(f_mrr)))
        res += ('f_hits1: {:.3f}\n'.format(np.mean(f_hits1)))
        res += ('f_hits5: {:.3f}\n'.format(np.mean(f_hits5)))
        res += ('f_hits10: {:.3f}\n'.format(np.mean(f_hits10)))
        res += ('==================================\n')
        print(res)
        print(res, file = self.metric_log, flush=True)
        sys.stdout.flush()

        # visdom
        mae_x = np.column_stack(( np.array([self.batch_nums]), np.array([self.batch_nums]), np.array([self.batch_nums]), np.array([self.batch_nums]) ))
        mae_y = np.column_stack(( np.array([np.mean(mae)]), np.array([np.mean(mean_confidence)]), np.array([np.mean(mean_var)]), np.array([np.mean(mean_neg_var)]) ))
        self.vis.line(mae_y, mae_x, self.win_mae, update="append") 
        hit_x = np.column_stack(( np.array([self.batch_nums]), np.array([self.batch_nums]), np.array([self.batch_nums]), np.array([self.batch_nums]) ))
        hit_y = np.column_stack(( np.array([np.mean(r_hits1)]), np.array([np.mean(r_hits10)]), np.array([np.mean(f_hits1)]), np.array([np.mean(f_hits10)]) ))
        self.vis.line(hit_y, hit_x, self.win_hit, update="append") 
        hit_x = np.column_stack(( np.array([self.batch_nums]), np.array([self.batch_nums]) ))
        hit_y = np.column_stack(( np.array([np.mean(r_mr)]), np.array([np.mean(r_mr)]) ))
        self.vis.line(hit_y, hit_x, self.win_mr, update="append") 

        if self.batch_nums == self.max_batches:
            nni.report_final_result(np.mean(f_hits1))
        else:
            nni.report_intermediate_result(np.mean(f_hits1))

        self.matcher.train()

        tot_time_e = time.time()
        tot_time = tot_time_e - tot_time_s
        print("[INFO] eval cost time:", tot_time)
        return np.mean(mean_confidence), np.mean(mean_var), np.mean(mae), np.mean(mse), np.mean(r_hits10), np.mean(r_hits5), np.mean(r_hits1), np.mean(r_mrr)


if __name__ == '__main__':
    args = read_args()

    # model execution 
    model_run = Model_Run(args)
    
    # train/test model
    if args.test:
        model_run.test()
    else:
        model_run.train()
