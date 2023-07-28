import subprocess
import h5py
import glob
import os


class RunCmd(object):
    def cmd_run(self, cmd):
        self.cmd = cmd
        subprocess.call(self.cmd, shell=True)


# Train the model
for i in range(3):
        print('train：',i)
        a = RunCmd()
        a.cmd_run('python MARSNet.py \
            --posi=./GraphProt_CLIP_sequences/ZC3H7B_Baltz2012.train.positives.fa \
            --nega=./GraphProt_CLIP_sequences/ZC3H7B_Baltz2012.train.negatives.fa\
            --model_type=ARSNet --train=True --n_epochs=50')





