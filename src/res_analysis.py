'''
Experiment result analysis tools
'''

from visdom import Visdom
import numpy as np
import json
# vis = None
# rel2triples = {} # [rel] = triples_info

def read_res(file):
    '''
    read triple file (h, r, t, s, r_rank, f_rank, mae, mse)
    '''
    triples_info = []
    with open(file) as f:
        lines = f.readlines()
        for line in lines:
            (h, r, t, s, r_rank, f_rank, mae, mse) = line.replace('\n', '').split('\t')
            triples_info.append((h, r, t, s, r_rank, f_rank, mae, mse)) # (h, r, t, s, r_rank, f_rank, mae, mse)
    f.close()   
    return triples_info 

def rel_2_triples(triples_info):
    '''
    create rel2triples
    '''
    rel2triples = {}
    triples = triples_info

    for triple in triples:
        rel = triple[1]
        if rel not in rel2triples.keys():
            rel2triples[rel] = []
        rel2triples[rel].append(triple)
    return rel2triples

def stati_by_rel(triples_info):
    '''
    statistic by rel
    '''
    rel2triples = rel_2_triples(triples_info)

    rel_rank, rel_mae, rel_num, rel_candidate_size = [], [], [], []
    for rel in rel2triples.keys():
        triples = rel2triples[rel]
        triples = np.array(triples)
        r_rank, f_rank, mae, mse = np.mean(triples[:,4].astype('int')), np.mean(triples[:,5].astype('int')), np.mean(triples[:,6].astype('float')), np.mean(triples[:,7].astype('float'))
        rel_rank.append([f_rank])
        rel_mae.append([mae])
        rel_num.append([triples.shape[0]])
        rel_candidate_size.append([len(rel2candidates[rel])])

    return np.array(rel_rank), np.array(rel_mae), np.array(rel_num), np.array(rel_candidate_size)

def workflow(args):
    
    global rel2candidates
    rel2candidates = json.load(open(args.datapath + '/rel2candidates_all.json'))
    vis = Visdom(env = args.experiment_name)
    res_file = './Experiments/' + args.experiment_name + '/result.log'

    triples_info = read_res(res_file)
    rel2triples = rel_2_triples(triples_info)
    rel_rank, rel_mae, rel_num, rel_candidate = stati_by_rel(triples_info)
    # print(rel_rank.shape, rel_candidate.shape)
    vis.bar(
        X = np.concatenate((rel_rank, rel_candidate), axis = 1), 
        opts=dict(
            title = 'rank',
            # stacked=True, 
            legend=['f_rank', 'candidate'], 
            rownames = list(rel2triples.keys())
        )
    )
    vis.bar(
        X = rel_mae,  
        opts=dict(
            title = 'mae',
            stacked=True, 
            # legend=['mae'], 
            rownames = list(rel2triples.keys())
        )
    )
    # vis.bar(
    #     X = rel_num, 
    #     opts=dict(
    #         title = 'query num',
    #         stacked=True, 
    #         # legend=['mae'], 
    #         rownames = list(rel2triples.keys())
    #     )
    # )

def compare_res(file_a, file_b):
    '''
    compare two result.log files
    '''
    vis = Visdom(env = 'Result Vs.')
    triples_info_a = read_res(file_a)
    rel2triples_a = rel_2_triples(triples_info_a)
    rel_rank_a, rel_mae_a, _, _ = stati_by_rel(triples_info_a)

    triples_info_b = read_res(file_b)
    rel2triples_b = rel_2_triples(triples_info_b)
    rel_rank_b, rel_mae_b, _, _ = stati_by_rel(triples_info_b)

    if set(rel2triples_a.keys()) != set(rel2triples_b.keys()):
        print('[WARN] Compare file relation not same')

    vis.bar(
        X = np.concatenate((rel_rank_a, rel_rank_b), axis = 1),  
        opts=dict(
            title = 'rank',
            # stacked = True, 
            legend = [file_a, file_b],
            rownames = list(rel2triples_a.keys())
        )
    )

    vis.bar(
        X = np.concatenate((rel_mae_a, rel_mae_b), axis = 1),  
        opts=dict(
            title = 'mae',
            # stacked = True, 
            legend = [file_a, file_b], 
            rownames = list(rel2triples_a.keys())
        )
    )

if __name__ == "__main__":
    # base_path = '../data/Experiment-4/' 
    # checkflow(base_path)
    compare_res('../Experiments/2_FSUKGE_R_N3/result.log', '../Experiments/2_FSUKGE_M_N0/result.log')
    # compare_res('../Experiments/2_FSUKGE_R_N3/result.log', '../Experiments/1_FSRL_MP_N3/result.log')
