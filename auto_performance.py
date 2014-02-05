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
    misgrasps_logged = True

    #for i_task in range(len(results_file)):
    for i_task in results_file:
        task_info = results_file[str(i_task)]
        knot_exists = False
        infeasible = False
        misgrasp = False

        for i_step in range(len(task_info)):
            step_info = task_info[str(i_step)]
            try:
                if step_info['misgrasp']:
                    misgrasp = True
                if step_info['infeasible']:
                    infeasible = True
            except:
                misgrasps_logged = False
                 
            try:
                rope_nodes = step_info['rope_nodes'][()]
            except:
                rope_nodes = step_info
            
            if isKnot(rope_nodes):
                num_knots += 1
                knot_exists = True
                break

        if not knot_exists:
            if infeasible:
                num_infeasible += 1
            elif misgrasp:
                num_misgrasps += 1
            print i_task
    
    if misgrasps_logged:
        print "# Misgrasps:", num_misgrasps
        print "# Infeasible:", num_infeasible
    return float(num_knots)/len(results_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("results_file", type=str)
    args = parser.parse_args()
    
    print "Success rate:", estimate_performance(args.results_file)
