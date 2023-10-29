import argparse


def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument('--phi1_path', 
                        type=str,
                        required=True,
                        help="Path to the embeddings in first representation space")

    parser.add_argument('--phi2_path',
                        type=str,
                        required=True,
                        help="Path to the embeddings in second representation space")

    parser.add_argument('--phi1_path_val',
                        type=str,
                        help="Path to the embeddings in first representation space to compute metrics."
                             " If not provided phi1_path will be also used for evaluation.")

    parser.add_argument('--phi2_path_val',
                        type=str,
                        help="Path to the embeddings in second representation space to compute metrics."
                             " If not provided phi2_path will be also used for evaluation.")

    parser.add_argument('--gt_labels_path',
                        type=str,
                        required=True,
                        help="Path to ground truth labeling to compute metrics")

    parser.add_argument('--k',
                        type=int,
                        default=10,
                        help="Number of classes")

    parser.add_argument('--inner_lr',
                        type=float,
                        default=0.001,
                        help="Step size for the inner optimization")

    parser.add_argument('--outer_lr',
                        type=float,
                        default=0.001,
                        help="Step size for the task encoder's updates")

    parser.add_argument('--tau',
                        type=float,
                        default=0.1,
                        help="Temperature hyperparameter")

    parser.add_argument('--H_reg',
                        type=float,
                        default=10.,
                        help="Entropy regularization coefficient")

    parser.add_argument('--num_iters',
                        type=int,
                        default=1000,
                        help="Number of training iterations")

    parser.add_argument('--adaptation_steps',
                        type=int,
                        default=300,
                        help="Number of inner iterations to fit linear model")

    parser.add_argument('--num_subsets',
                        type=int,
                        default=20,
                        help="Number of (Xtr, Xte) subsets for averaging HUME's loss")

    parser.add_argument('--subset_size',
                        type=int,
                        default=10000,
                        help="Size of union of each (Xtr, Xte) subset")

    parser.add_argument('--train_fraction',
                        type=float,
                        default=0.9,
                        help="Fraction of args.subset_size to define size of Xtr")

    parser.add_argument('--no_anneal',
                        dest='anneal',
                        action='store_false',
                        help="Turn off temperature and learning rate annealing")

    parser.add_argument('--no_rand_init',
                        dest='rand_init',
                        action='store_false',
                        help="Start from random inner w0 at each outer iter or generate random w0 once")

    parser.add_argument('--device',
                        type=str,
                        default="cuda",
                        help="Use cuda or cpu")

    parser.add_argument('--exp_path',
                        type=str,
                        default="./linear_tasks/",
                        help="Path to save experiment's results")

    parser.add_argument('--save_all',
                        action='store_true',
                        help="If used then task_encoder is saved at each iteration")

    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help='Random seed')
    
    return parser.parse_args(args)
