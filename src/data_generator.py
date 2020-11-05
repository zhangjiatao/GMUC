import json
import random

def train_generate(datapath, neg_nums , batch_size, few, symbol2id, ent2id, e1rel_e2, type_constrain):
    '''
    create train data
    '''
    # # 5_FSRL_T3-2_query
    # train_tasks_n0 = json.load(open(datapath + '/train_tasks_n0.json'))

    train_tasks = json.load(open(datapath + '/train_tasks.json'))
    rel2candidates = json.load(open(datapath + '/rel2candidates_all.json'))
    task_pool = list(train_tasks.keys())
    #print (task_pool[0])

    num_tasks = len(task_pool)

    # for query_ in train_tasks.keys():
    # 	print len(train_tasks[query_])
    # 	if len(train_tasks[query_]) < 4:
    # 		print len(train_tasks[query_])

    print ("train data generation")

    rel_idx = 0

    while True:
        if rel_idx % num_tasks == 0:
            random.shuffle(task_pool)
        query = task_pool[rel_idx % num_tasks]
        #print (query)
        rel_idx += 1

        #query_rand = random.randint(0, (num_tasks - 1))
        #query = task_pool[query_rand]

        candidates = rel2candidates[query]
        #print rel_idx

        if len(candidates) <= 20:
            continue

        train_and_test = train_tasks[query]
        random.shuffle(train_and_test)

        support_triples = train_and_test[:few]
        support_pairs = [[symbol2id[triple[0]], symbol2id[triple[2]], float(triple[3])] for triple in support_triples] # (h, t, s)
        support_left = [ent2id[triple[0]] for triple in support_triples]
        support_right = [ent2id[triple[2]] for triple in support_triples]

        all_test_triples = train_and_test[few:]

        # #start 5_FSRL_T3-2_query
        # all_test_triples_n0 = []
        # train_and_test_n0 = train_tasks_n0[query]
        # random.shuffle(train_and_test_n0)
        # for triple in train_and_test_n0:
        #     if triple not in support_triples:
        #         all_test_triples_n0.append(triple)
        # all_test_triples = all_test_triples_n0
        # #end 5_FSRL_T3-2_query

        if len(all_test_triples) == 0:
            continue

        if len(all_test_triples) < batch_size:
            query_triples = [random.choice(all_test_triples) for _ in range(batch_size)]
        else:
            query_triples = random.sample(all_test_triples, batch_size)

        query_pairs = [[symbol2id[triple[0]], symbol2id[triple[2]], float(triple[3])] for triple in query_triples] # (h, t, s)
        query_left = [ent2id[triple[0]] for triple in query_triples]
        query_right = [ent2id[triple[2]] for triple in query_triples]
        query_confidence = [float(triple[3]) for triple in query_triples]

        false_pairs = []
        false_left = []
        false_right = []
        if not type_constrain:
            candidates = list(ent2id.keys()) # add all entity
        for triple in query_triples:
            for i in range(neg_nums):
                e_h = triple[0]
                rel = triple[1]
                e_t = triple[2]
                while True:
                    noise = random.choice(candidates)
                    if (noise not in e1rel_e2[e_h+rel]) and noise != e_t:
                        break
                false_pairs.append([symbol2id[e_h], symbol2id[noise], 0.0]) # (h, t, s)
                false_left.append(ent2id[e_h])
                false_right.append(ent2id[noise])

        yield support_pairs, query_pairs, false_pairs, support_left, support_right, query_left, query_right, query_confidence, false_left, false_right


