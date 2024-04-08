import os, shutil
import numpy as np
from glob import glob

all_spec = glob('spectra/*.ascii')

def clean_copy(src, dst):
    """copy a file, deleting the target if it already exists. If target is a directory,
    preserve the original file name, otherwise move change name.

    Parameters
    ----------
    src: string
        Name of the source file.

    dst: string
        Name of the destination file or directory.

    """
    if os.path.isdir(dst): #assume copying same file to different directory
        if os.path.exists(os.path.join(dst,src)):
            os.remove(os.path.join(dst,src))
    else:
        if os.path.exists(dst):
            os.remove(dst)
    shutil.copy(src, dst)
    return


#read in ids from references file
ref_ids = []
with open('lasd_measurements.cat','r') as mfile:
    for line in mfile:
        temp = line.strip().split(None)
        if temp[0] != '#':
            ref_ids.append(temp[0])

#iterate over each measurement id and see if I can find a matching spec file
for rid in ref_ids:
    found = 0
    for spec in all_spec:
        spec_spot = spec.find(rid)
        if spec_spot >= 0:
            print(rid, spec)
            found = 1
            clean_copy(spec, '{0}.ascii'.format(rid))
            break
    if not found:
        print(rid, 'no match')
