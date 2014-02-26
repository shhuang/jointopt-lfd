#!/usr/bin/env python
# Assumes structure of h5 files are the same

import argparse, h5py, numpy as np

def report_differences_for_branch(results1, results2, name):
    if type(results1[name]) != h5py._hl.group.Group:
        val1 = results1[name][()]
        if name not in results2:
            print "WARNING: Branch {0} missing in second file".format(name)
            return
        val2 = results2[name][()]

        if type(val1) == str:
            if val1 != val2:
                print "{0} is different: {1} vs. {2}".format(name, val1, val2)
        elif type(val1) in [np.int64, np.float64]:
            if val1 != val2:
                print "{0} is different: val1 - val2 = {1}".format(name, val1 - val2)
        elif type(val1) == np.ndarray:
            if val1.shape != val2.shape:
                print "{0} is different: {1} vs. {2} shape".format(name, val1.shape, val2.shape)
            elif not (val1==val2).all():
                print "{0} is different: ||val1 - val2||_F = {1}".format(name, np.linalg.norm(val1 - val2))
        elif val1 != val2:
            print "{0} is different: {1} vs. {2}".format(name, val1, val2)
        return
    for k in results1[name].keys():
        report_differences_for_branch(results1, results2, results1[name][k].name)

def report_missing_for_branch(results1, results2, name):
    if type(results2[name]) != h5py._hl.group.Group:
        if name not in results1:
            print "WARNING: Branch {0} missing in first file".format(name)
        return
    for k in results2[name].keys():
        report_missing_for_branch(results1, results2, results2[name][k].name)

def report_differences(results_file1, results_file2):
    results1 = h5py.File(results_file1, 'r')
    results2 = h5py.File(results_file2, 'r')
    for task_id in results1:
        if task_id not in results2:
            print "WARNING: Task id {0} missing in {1}".format(task_id, results_file2)
        report_differences_for_branch(results1, results2, results1[task_id].name)

    # Check for tasks in results2 not but results1
    for task_id in results2:
        if task_id not in results1:
            print "WARNING: Task id {0} missing in {1}".format(task_id, results_file1)
        report_missing_for_branch(results1, results2, results2[task_id].name)

    results1.close()
    results2.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_file1", type=str)
    parser.add_argument("results_file2", type=str)

    args = parser.parse_args()

    report_differences(args.results_file1, args.results_file2)
