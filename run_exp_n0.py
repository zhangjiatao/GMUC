from src.model_run import Model_Run
from src.args import read_args
from src.input_analysis import checkflow
from src.res_analysis import workflow
import os

if __name__ == '__main__':
    args = read_args()


    args.experiment_name = 'GMUC_N0_conf' # Experiment ID
    args.set_aggregator = 'GMUC' 
    args.datapath = './data/MAGA-PLUS-NL27K/NL27K-N0'
    args.eval_every = 5000
    args.max_batches = 60000
    args.if_conf = 1
    # args.rank_weight = 0.0
    # args.ae_weight = 0.0

    # make Experiment dir
    exp_path = './Experiments/' + args.experiment_name
    if(os.path.exists(exp_path) == False):
        os.makedirs(exp_path)
    if(os.path.exists(exp_path + '/checkpoints') == False):
        os.makedirs(exp_path + '/checkpoints')

    # input dataset analysis
    # checkflow(args)

    # model execution 
    model_run = Model_Run(args)
    model_run.train()

    # result analysis
    workflow(args)
