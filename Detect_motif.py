import subprocess
class RunCmd(object):
  def cmd_run(self, cmd):
    self.cmd = cmd
    subprocess.call(self.cmd, shell=True)

for i in range(1):
    a = RunCmd()
    a.cmd_run('python MARSNet.py \
    --posi=./GraphProt_CLIP_sequences/ZC3H7B_Baltz2012.train.positives.fa \
    --nega=./GraphProt_CLIP_sequences/ZC3H7B_Baltz2012.train.negatives.fa \
    --motif=True  --n_epochs=50')

