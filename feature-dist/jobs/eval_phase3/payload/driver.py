import os
import glob
import subprocess

if __name__ == '__main__':
    currdir = os.path.dirname(os.path.realpath(__file__))
    logdir = os.path.join(currdir, 'logs')
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    outdir = os.path.join(currdir, 'out')
    if not os.path.exists(outdir):
        os.mkdir(outdir)
#    outimagedir = os.path.join(outdir, 'images')
#    if not os.path.exists(outimagedir):
#        os.makedirs(outimagedir)
    os.chdir(currdir)
    subprocesses = []
    for fname in glob.glob('splits/*.h5'):
        fname = os.path.join(currdir, fname)
        _, nameonly = os.path.split(fname)
        logname = os.path.join(logdir, nameonly + '.txt')
        errname = os.path.join(logdir, nameonly + '.err')
        outname = os.path.join(outdir, nameonly)
        subprocesses.append(subprocess.Popen('python do_task_eval.py --elbow_obstacle --animation 0 --resultfile={} data/misc/actions.h5 {} regcost-trajopt'.format(outname, fname),
                            stdout=open(logname, 'w'),
                            stderr=open(errname, 'w'),
                            shell=True))
#        subprocesses.append(subprocess.Popen('python holdout_result.py {} {}'.format(outname, outimagedir),
#                            stdout=open(logname, 'w'),
#                            stderr=open(errname, 'w'),
#                            shell=True))
    exit_codes = [p.wait() for p in subprocesses]

