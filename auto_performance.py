#!/usr/bin/env python

import argparse
import h5py
from knot_classifier import isKnot

def estimate_performance(results_file):
    if type(results_file) is str:
        results_file = h5py.File(results_file, 'r')

    num_knots = 0
    num_misgrasps = 0
    num_infeasible = 0
    action_time = 0
    exec_time = 0
    misgrasps_logged = True
    timing_logged = True

    for (i_task, task_info) in sorted(results_file.iteritems(), key=lambda item: int(item[0])):
        knot_exists = False
        infeasible = False
        misgrasp = False

        for i_step in range(len(task_info) - (1 if 'init' in task_info else 0)):
            step_info = task_info[str(i_step)]
            try:
                if step_info['misgrasp'][()]:
                    misgrasp = True
                if step_info['infeasible'][()]:
                    infeasible = True
            except:
                misgrasps_logged = False
                 
            try:
                rope_nodes = step_info['rope_nodes'][()]
            except:
                rope_nodes = step_info
            
            if isKnot(rope_nodes):
                knot_exists = True
                break

            try:
                action_time += step_info['action_time'][()]
                exec_time += step_info['exec_time'][()]
            except:
                timing_logged = False

        if infeasible and knot_exists:
            print 'infeasible but ended up in knot'
            knot_exists = False

        if not knot_exists:
            if infeasible:
                num_infeasible += 1
            elif misgrasp:
                num_misgrasps += 1
            print i_task

        if knot_exists:
            num_knots += 1
    
    if misgrasps_logged:
        print "# Misgrasps:", num_misgrasps
        print "# Infeasible:", num_infeasible
    if timing_logged:
        print "Time taken to choose demo:", action_time, "seconds"
        print "Time taken to warp and execute demo:", exec_time, "seconds"
    return num_knots

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("results_file", type=str)
    args = parser.parse_args()
    
    results_file = h5py.File(args.results_file, 'r')
    
    num_successes = estimate_performance(args.results_file)
    print "Successes / Total: %d/%d" % (num_successes, len(results_file))
    print "Success rate:", float(num_successes)/float(len(results_file))
