import argparse

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--set_aggregator", default="FSRL", type=str) # model ['FSUKGE', 'FSRL']
    parser.add_argument("--datapath", default="../data/NL27K-FSUKGE", type=str)
    parser.add_argument("--random_seed", default=1, type=int)
    parser.add_argument("--random_embed", default=1, type=int)
    parser.add_argument("--few", default=3, type=int)
    parser.add_argument("--test", default=0, type=int) # 0 train, 1 test
    parser.add_argument("--embed_model", default='ComplEx', type=str)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--embed_dim", default=100, type=int)
    parser.add_argument("--dropout", default=0.5, type=float)
    parser.add_argument("--fine_tune", default=0, type=int)
    # parser.add_argument("--aggregate", default='max', type=str)
    parser.add_argument("--process_steps", default=2, type=int) # Queryencoder
    # parser.add_argument("--aggregator", default='max', type=str)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--weight_decay", default=0, type=float)
    parser.add_argument("--max_neighbor", default=30, type=int)
    parser.add_argument("--train_few", default=1, type=int)
    parser.add_argument("--margin", default=5.0, type=float)
    parser.add_argument("--eval_every", default=5000, type=int)
    parser.add_argument("--max_batches", default=40000, type=int)
    # parser.add_argument("--prefix", default='intial', type=str)

    parser.add_argument("--rank_weight", default=1.0, type=float)
    parser.add_argument("--ae_weight", default=0.00001, type=float)
    parser.add_argument("--mae_weight", default=1.0, type=float)
    parser.add_argument("--if_conf", default=1, type=int) # if consider triple confidence

    parser.add_argument("--if_GPU", default=1, type=int)
    parser.add_argument("--type_constrain", default=1, type=int) # if type_constrain
    parser.add_argument("--neg_nums", default = 1, type=int) # neg num for each train query
    parser.add_argument("--sim", default = 'KL', type=str) # [EL, KL] KG2E similair function
    parser.add_argument("--experiment_name", default = 'default', type=str)

    args = parser.parse_args()

    return args

