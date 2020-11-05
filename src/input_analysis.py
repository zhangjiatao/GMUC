# encoding: utf-8
'''
data set anaysis
'''

import json
import numpy as np
import os

exp_path = None

def write_triples(file, triples):
    '''
    write triples file
    '''
    with open(file, 'w') as f:
        for triple in triples:
            line = str(triple[0]) + '\t' + str(triple[1]) + '\t' + str(triple[2]) + '\t' + str(triple[3]) + '\n'
            f.write(line)
    f.close()

def read_triples(file):
    '''
    read triple file (h, r, t, s)
    '''
    triples = []
    with open(file) as f:
        lines = f.readlines()
        for line in lines:
            (h, r, t, s) = line.replace('\n', '').split('\t')
            triples.append((h, r, t, s)) # (h, r, t, s)
    f.close()   
    return triples    

def tasks_2_triples(task_file):
    '''
    read task file and return triples
    '''
    triples = []
    tasks = list(json.load(open(task_file)).values())
    for task in tasks:
        for triple in task:
            triples.append(triple)
    return triples

def triples_2_str_set(triples):
    '''
    triple_list to set
    '''
    triple_set = set()
    for triple in triples:
        (h, r, t, s) = triple
        tmp_str = h + ',' + r + ',' + t
        triple_set.add(tmp_str)
    return triple_set

def find_neg(triples_n0, triples_nx):
    '''
    find neg triples by compare with N0
    '''
    triples_neg = []
    triples_n0_set = triples_2_str_set(triples_n0)
    for triple in triples_nx:
        (h, r, t, s) = triple
        tmp_str = h + ',' + r + ',' + t
        if tmp_str not in triples_n0_set:
            triples_neg.append((h, r, t, s))
    return triples_neg

def mean_and_std(triples, desc):
    '''
    get mean and std of data
    '''
    if len(triples) == 0:
        return
    scores = np.array(triples)[:,3].astype('float')
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    tmp_str = 'Num: %06d | Max: %.3lf | Min: %.3lf | Mean: %.3lf | Std: %.3lf | Desc: %s' % (scores.shape[0], scores.max(), scores.min(), mean_score, std_score, desc)
    print(tmp_str)
    print(tmp_str, file = exp_path, flush = True)
    # print('Num: %06d | Max: %.3lf | Min: %.3lf | Mean: %.3lf | Std: %.3lf | Desc: %s' % (scores.shape[0], scores.max(), scores.min(), mean_score, std_score, desc))

def stati_dataset(train, dev, test, path, desc):
    print('---------%s----------' % desc)
    print('---------%s----------' % desc, file = exp_path, flush = True)
    triples_all = train + dev + test + path
    mean_and_std(triples_all, 'dataset overall')
    mean_and_std(train, 'train task all(neg rage {:.3f})'.format( (len(train) - len(train_triples_n0)) / len(train_triples_n0) ))
    mean_and_std(dev, 'dev task all(neg rage {:.3f})'.format( (len(dev) - len(dev_triples_n0)) / len(dev_triples_n0) ))
    mean_and_std(test, 'test task all(neg rage {:.3f})'.format( (len(test) - len(test_triples_n0)) / len(test_triples_n0) ))
    mean_and_std(path, 'graph path all(neg rage {:.3f})\n'.format( (len(path) - len(path_triples_n0)) / len(path_triples_n0) ))

    neg_triples = find_neg(train_triples_n0, train)
    mean_and_std(neg_triples, 'train neg triples')
    neg_triples = find_neg(dev_triples_n0, dev)
    mean_and_std(neg_triples, 'dev neg triples')
    neg_triples = find_neg(test_triples_n0, test)
    mean_and_std(neg_triples, 'test neg triples')
    neg_triples = find_neg(path_triples_n0, path)
    mean_and_std(neg_triples, 'path neg triples')

def analysis_train(base_path):
    '''
    statistic the negtive examples
    '''
    global train_triples_n0, dev_triples_n0, test_triples_n0, path_triples_n0

    print('==================== Data Statistics ====================')
    print('==================== Data Statistics ====================', file = exp_path, flush = True)

    # -------------------- check ----------------------
    print('[INFO] DataSet %s' % base_path)
    print('[INFO] DataSet %s' % base_path, file = exp_path, flush = True)
    # read train triples
    train_triples_n0 = tasks_2_triples(base_path + '/NL27K-N0/train_tasks.json')
    train_triples_n1 = tasks_2_triples(base_path + '/NL27K-N1/train_tasks.json')
    train_triples_n2 = tasks_2_triples(base_path + '/NL27K-N2/train_tasks.json')
    train_triples_n3 = tasks_2_triples(base_path + '/NL27K-N3/train_tasks.json')

    # relation check
    train_rels_n0 = json.load(open(base_path + '/NL27K-N0/train_tasks.json')).keys()
    train_rels_n1 = json.load(open(base_path + '/NL27K-N1/train_tasks.json')).keys()
    train_rels_n2 = json.load(open(base_path + '/NL27K-N2/train_tasks.json')).keys()
    train_rels_n3 = json.load(open(base_path + '/NL27K-N3/train_tasks.json')).keys()
    if train_rels_n0 == train_rels_n1 and train_rels_n0 == train_rels_n2 and train_rels_n0 == train_rels_n3:
        print('[INFO] train tasks relations check ok')
        print('[INFO] train tasks relations check ok', file = exp_path, flush = True)
    else:
        print('[WARN] train tasks relations NOT OK ！！！')
        print('[WARN] train tasks relations NOT OK ！！！', file = exp_path, flush = True)

    test_triples_n0 = tasks_2_triples(base_path + '/NL27K-N0/test_tasks.json')
    test_triples_n1 = tasks_2_triples(base_path + '/NL27K-N1/test_tasks.json')
    test_triples_n2 = tasks_2_triples(base_path + '/NL27K-N2/test_tasks.json')
    test_triples_n3 = tasks_2_triples(base_path + '/NL27K-N3/test_tasks.json')
    if triples_2_str_set(test_triples_n0) == triples_2_str_set(test_triples_n1) and triples_2_str_set(test_triples_n0) == triples_2_str_set(test_triples_n2) and triples_2_str_set(test_triples_n0) == triples_2_str_set(test_triples_n3):
        print('[INFO] test tasks triples check ok')
        print('[INFO] test tasks triples check ok', file = exp_path, flush = True)
    else:
        print('[WARN] test tasks triples NOT OK ！！！')
        print('[WARN] test tasks triples NOT OK ！！！', file = exp_path, flush = True)

    dev_triples_n0 = tasks_2_triples(base_path + '/NL27K-N0/dev_tasks.json')
    dev_triples_n1 = tasks_2_triples(base_path + '/NL27K-N1/dev_tasks.json')
    dev_triples_n2 = tasks_2_triples(base_path + '/NL27K-N2/dev_tasks.json')
    dev_triples_n3 = tasks_2_triples(base_path + '/NL27K-N3/dev_tasks.json')
    if triples_2_str_set(dev_triples_n0) == triples_2_str_set(dev_triples_n1) and triples_2_str_set(dev_triples_n0) == triples_2_str_set(dev_triples_n2) and triples_2_str_set(dev_triples_n0) == triples_2_str_set(dev_triples_n3):
        print('[INFO] dev tasks triples check ok')
        print('[INFO] dev tasks triples check ok', file = exp_path, flush = True)
    else:
        print('[WARN] dev tasks triples NOT OK ！！！')
        print('[WARN] dev tasks triples NOT OK ！！！', file = exp_path, flush = True)

    path_triples_n0 = read_triples(base_path + '/NL27K-N0/path_graph')
    path_triples_n1 = read_triples(base_path + '/NL27K-N1/path_graph')
    path_triples_n2 = read_triples(base_path + '/NL27K-N2/path_graph')
    path_triples_n3 = read_triples(base_path + '/NL27K-N3/path_graph')
    if triples_2_str_set(path_triples_n0) == triples_2_str_set(path_triples_n1) and triples_2_str_set(path_triples_n0) == triples_2_str_set(path_triples_n2) and triples_2_str_set(path_triples_n0) == triples_2_str_set(path_triples_n3):
        print('[INFO] path triples check ok')
        print('[INFO] path triples check ok', file = exp_path, flush = True)
    else:
        print('[WARN] path triples NOT OK ！！！')
        print('[WARN] path triples NOT OK ！！！', file = exp_path, flush = True)

    # -------------------- statistic ----------------------
    stati_dataset(train_triples_n0, dev_triples_n0, test_triples_n0, path_triples_n0, 'N0')
    stati_dataset(train_triples_n1, dev_triples_n1, test_triples_n1, path_triples_n1, 'N1')
    stati_dataset(train_triples_n2, dev_triples_n2, test_triples_n2, path_triples_n2, 'N2')
    stati_dataset(train_triples_n3, dev_triples_n3, test_triples_n3, path_triples_n3, 'N3')


def checkflow(args):
    global exp_path
    data_path = args.datapath
    data_path = os.path.dirname(data_path) 
    exp_path = './Experiments/' + args.experiment_name + '/DataSet.log' # output dir
    exp_path = open(exp_path, 'w')
    analysis_train(data_path)

if __name__ == "__main__":
    base_path = '../data/Experiment-4/' 
    checkflow(base_path)