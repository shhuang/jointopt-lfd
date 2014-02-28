#!/usr/bin/env python

import argparse, h5py

DEFAULT_TASKS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]

def subselect_holdout(task_list, holdout_file, output_file):
    tasks = DEFAULT_TASKS if task_list is None else task_list
    holdout_tasks = h5py.File(holdout_file, 'r')
    output_holdout = h5py.File(output_file, 'w')
    for (i, t) in enumerate(tasks):
        print "Copying over task", i
        if str(t) not in holdout_tasks:
            continue
        g = output_holdout.create_group(str(i))
        for k in holdout_tasks[str(t)].keys():
            g[k] = holdout_tasks[str(t)][k][()]
    holdout_tasks.close()
    output_holdout.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("orig_holdout_file", type=str)
    parser.add_argument("output_holdout_file", type=str)
    parser.add_argument("--tasks", nargs='+', type=int)
    parser.add_argument("--dummy", action="store_true")  # Required after specifying --tasks, to avoid parsing error
    args = parser.parse_args()

    subselect_holdout(args.tasks, args.orig_holdout_file, args.output_holdout_file)
