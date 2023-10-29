import os
import re
import pickle
import argparse

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn

from metrics import cluster_acc, cluster_ari


def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument('--phi1_path', 
                        type=str,
                        required=True,
                        help="Path to the embeddings in first representation space")

    parser.add_argument('--tasks_path',
                        type=str,
                        required=True,
                        help="Path to tasks to evaluate")

    parser.add_argument('--gt_labels_path',
                        type=str,
                        required=True,
                        help="Path to ground truth labels")
    
    return parser.parse_args(args)


def match_tasks(tasks, cv_scores):
    idx = np.argmax(cv_scores)
    tasks_matched = []
    for i in range(len(cv_scores)):
        if i == idx:
            tasks_matched.append(tasks[idx])
        else:
            cacc, task_match = cluster_acc(tasks[i], tasks[idx], return_matched=True)
            tasks_matched.append(task_match)
    tasks_matched = np.array(tasks_matched)
    return tasks_matched

    
def get_maj_vote(tasks):
    axis = 0
    u, indices = np.unique(tasks, return_inverse=True)
    result = u[np.argmax(np.apply_along_axis(np.bincount, axis, indices.reshape(tasks.shape), None, np.max(indices) + 1), axis=axis)]
    return result


def get_fnames(path):
    task_pattern = r'linear_task_(\d+)\.pt'
    results_pattern = r'results_(\d+)\.pickle'

    tasks_fnames = []
    results_fnames = []

    for filename in os.listdir(path):
        task_match = re.match(task_pattern, filename)
        results_match = re.match(results_pattern, filename)
        if task_match:
            tasks_fnames.append(filename)
        elif results_match:
            results_fnames.append(filename)

    extract_i = lambda filename: int(re.search(r'\d+', filename).group())

    tasks_fnames = sorted(tasks_fnames, key=extract_i)
    results_fnames = sorted(results_fnames, key=extract_i)
    return tasks_fnames, results_fnames


def run(args=None):
    args=parse_args(args)

    phi1 = np.load(args.phi1_path).astype(np.float32)
    y_true = np.load(args.gt_labels_path)
    k = len(np.unique(y_true))

    tasks_fnames, results_fnames = get_fnames(args.tasks_path)

    if len(results_fnames) > 0:
        cv_scores = []
        for fname in results_fnames:
            with open(os.path.join(args.tasks_path, fname), "rb") as handle:
                cv_scores.append(pickle.load(handle)["CV_Score"])
        cv_scores = np.array(cv_scores)
    else:
        cv_scores = np.zeros(len(tasks_fnames))
        cv_scores[0] = 1.0

    assert len(cv_scores) == len(tasks_fnames)

    tasks = []
    device = "cpu"
    for fname in tqdm(tasks_fnames):
        state_dict = torch.load(os.path.join(args.tasks_path, fname), map_location="cpu")
        task_encoder = nn.utils.parametrizations.orthogonal(
            nn.Linear(phi1.shape[1], k, bias=False)
        )
        task_encoder.load_state_dict(state_dict)
        task = task_encoder(torch.from_numpy(phi1)).detach().numpy().argmax(1)
        tasks.append(task)
    tasks = np.array(tasks)
    tasks_matched = match_tasks(tasks, cv_scores)

    print(f"ACC: {cluster_acc(get_maj_vote(tasks_matched), y_true) * 100:.1f}")
    print(f"ARI: {cluster_ari(get_maj_vote(tasks_matched), y_true) * 100:.1f}")



if __name__ == '__main__':
    run()
