import numpy as np


with open('lasd_measurements.cat','w') as ofile:
    with open('zsysdf.ascii','r') as file:
        for ii, line in enumerate(file):
            temp = line.strip().split(None)
            if ii == 0:
                ids = temp
                ofile.write('# {0} {1} {2}\n'.format(ids[0], ids[58], ids[-4]))
            else:
                
                ofile.write('{0} {1:.5f} {2:.6f}\n'.format(temp[0], np.log10(float(temp[58])), float(temp[-4])))
        
