import os
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import learn2learn as l2l
import numpy as np
from tqdm import tqdm

from argparser import parse_args
from activations import Sparsemax
from utils import fix_seed, get_cv_score, check_both_none_or_not_none
from metrics import cluster_acc, cluster_ari


def run(args=None):
    args = parse_args(args)
    device = torch.device(args.device)
    fix_seed(args.seed)
    if not os.path.exists(args.exp_path):
        os.makedirs(args.exp_path)
       
    phi1 = np.load(args.phi1_path).astype(np.float32)
    phi2 = np.load(args.phi2_path).astype(np.float32)
    assert check_both_none_or_not_none(args.phi1_path_val, args.phi2_path_val)
    if args.phi1_path_val is not None:
        phi1_val = np.load(args.phi1_path_val).astype(np.float32)
        phi2_val = np.load(args.phi2_path_val).astype(np.float32)
    else:
        phi1_val = np.copy(phi1)
        phi2_val = np.copy(phi2)
    y_true_val = np.load(args.gt_labels_path)
    assert phi1.shape[0] == phi2.shape[0]
    assert phi1_val.shape[0] == phi2_val.shape[0]
    assert phi1_val.shape[0] == y_true_val.shape[0]
    n_train = phi1.shape[0]
    d1, d2 = phi2.shape[1], phi1.shape[1]
    subset_size = min(n_train, args.subset_size)
    
    # Instantiate linear layer for the inner optimization (Equation 5)
    inner_linear = nn.Linear(d1, args.k, bias=True).to(device)
    inner_linear = l2l.algorithms.MAML(inner_linear, lr=args.inner_lr)
    
    # Instantiate task encoder with orthogonal weights parametrization (Equation 3)
    task_encoder = nn.Linear(d2, args.k, bias=False).to(device)
    task_encoder = nn.utils.parametrizations.orthogonal(task_encoder)

    all_parameters = list(task_encoder.parameters())
    optimizer = torch.optim.Adam(all_parameters, lr=args.outer_lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[100, 200],
        gamma=0.1 if args.anneal else 1.0
    )
    old_lr = args.outer_lr
    tau = args.tau
    sparsemax_act = Sparsemax(dim=1)
    for i in tqdm(range(args.num_iters)):
        optimizer.zero_grad()
        mean_train_error = 0.0
        mean_valid_error = 0.0
        mean_valid_acc = 0.0
        mean_train_acc = 0.0
        mean_label_dist = 0.0
        mean_sparsity = 0.0

        for j in range(args.num_subsets):
            # Sample X_tr and X_te
            subset = np.random.choice(n_train, size=subset_size, replace=False)
            subset_tr = subset[:int(subset_size * args.train_fraction)]
            subset_te = subset[int(subset_size * args.train_fraction):]

            phi1_tr = torch.from_numpy(phi1[subset_tr]).to(device)
            phi1_te = torch.from_numpy(phi1[subset_te]).to(device)
            phi2_tr = torch.from_numpy(phi2[subset_tr]).to(device)
            phi2_te = torch.from_numpy(phi2[subset_te]).to(device)

            # Get labels using current task encoder
            task_labels_tr = sparsemax_act(task_encoder(phi1_tr) / tau)
            task_labels_te = sparsemax_act(task_encoder(phi1_te) / tau)
            task_labels_all = torch.cat((task_labels_tr, task_labels_te))
            
            """
            Perform inner optimization from the random initialization or 
            from fixed w0 (corresponds to Cold Start BLO for Equation 5)
            """

            if args.rand_init:
                inner_linear.reset_parameters()
            learner = inner_linear.clone()

            for step in range(args.adaptation_steps):
                train_error = F.cross_entropy(learner(phi2_tr), task_labels_tr)
                learner.adapt(train_error)

            # Compute HUME's objective (Equation 7)
            label_dist = task_labels_all.mean(0)
            entr = torch.special.entr(label_dist)
            valid_error = F.cross_entropy(learner(phi2_te), task_labels_te)

            # Accumulate gradients across args.num_subsets
            (valid_error - args.H_reg * entr.sum()).backward()

            # Compute training stats
            mean_train_error += train_error.item()
            mean_train_acc += torch.eq(
                learner(phi2_tr).argmax(1),
                task_labels_tr.argmax(1)
            ).float().mean().item()
            mean_valid_error += valid_error.item()
            mean_valid_acc += torch.eq(
                learner(phi2_te).argmax(1),
                task_labels_te.argmax(1)
            ).float().mean().item()
            mean_label_dist += label_dist.detach().cpu().numpy()
            mean_sparsity += task_labels_all[torch.arange(task_labels_all.shape[0]),
                                             task_labels_all.argmax(1)].mean().item()

        # Average gradients over args.num_subsets and update the task encoder parameters
        for p in all_parameters:
            p.grad.data.mul_(1.0 / args.num_subsets)
            print(f"Grad norm: {torch.norm(p.grad.data).item()}")
        nn.utils.clip_grad_norm_(task_encoder.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Anneal step size and temperature
        if scheduler.get_last_lr()[0] != old_lr:
            print("Annealed Learning rate")
            old_lr = scheduler.get_last_lr()[0]
            print("Annealed Temperature")
            tau = tau / 10
            print()
        
        # Print train stats
        print("Train stats:")
        print(f"Mean TrainError {mean_train_error / args.num_subsets}")
        print(f"Mean ValidError {mean_valid_error / args.num_subsets}")
        print(f"Mean TrainAcc {mean_train_acc / args.num_subsets}")
        print(f"Mean ValidAcc {mean_valid_acc / args.num_subsets}")
        print(f"Mean Sparsity {mean_sparsity / args.num_subsets}")
        print("Mean Label Dist:", mean_label_dist / args.num_subsets)
        print()

        # Print val stats
        out_all_val = task_encoder(torch.from_numpy(phi1_val).to(device))
        preds_all_val = torch.argmax(out_all_val, dim=1).detach().cpu().numpy()
        print("Val metrics:")
        print("Num found clusters:", len(np.unique(preds_all_val)))
        print(f"Cluster ACC epoch {i}:", cluster_acc(preds_all_val, y_true_val))
        print(f"Cluster ARI epoch {i}:", cluster_ari(preds_all_val, y_true_val))
        print()

        if args.save_all:
            torch.save(task_encoder.state_dict(), args.exp_path + f"linear_task_{i}_{args.seed}.pt")
            

    # Compute cross-validation accuracy w.r.t. found task and save the results
    out_all_val = task_encoder(torch.from_numpy(phi1_val).to(device))
    task_val = torch.argmax(out_all_val, dim=1).detach().cpu().numpy()
    final_cv_score = get_cv_score(phi2_val, task_val)
    with open(args.exp_path + f"results_{args.seed}.pickle", "wb") as handle:
        pickle.dump({"CV_Score": final_cv_score}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    torch.save(task_encoder.state_dict(), args.exp_path + f"linear_task_{args.seed}.pt")
    
if __name__ == '__main__':
    run()
