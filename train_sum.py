import subprocess
import h5py
import glob
import os


class RunCmd(object):
    def cmd_run(self, cmd):
        self.cmd = cmd
        subprocess.call(self.cmd, shell=True)


# Train the model three times and model predicts the result ten times on the test set

for b in range(1,25):
    if b == 1:
        print(b)
        all = glob.glob("./model*")
        for h in all:
            print(h)
            os.remove(h)
        for i in range(1,4):
            print('train：',i)
            a = RunCmd()
            a.cmd_run('python MARSNet.py \
               --posi=./GraphProt_CLIP_sequences/ALKBH5_Baltz2012.train.positives.fa \
               --nega=./GraphProt_CLIP_sequences/ALKBH5_Baltz2012.train.negatives.fa\
               --model_type=DRSN --train=True --n_epochs=50')
            for j in range(10):
                a = RunCmd()
                a.cmd_run('python MARSNet.py \
                   --testfile=./GraphProt_CLIP_sequences/ALKBH5_Baltz2012.ls.positives.fa \
                   --nega=./GraphProt_CLIP_sequences/ALKBH5_Baltz2012.ls.negatives.fa\
                   --model_type=DRSN --predict=True')

    elif b == 2:
        print(b)
        all = glob.glob("./model*")
        for h in all:
            print(h)
            os.remove(h)

        for i in range(1,4):
            print('train：',i)
            a = RunCmd()
            a.cmd_run('python MARSNet.py \
            --posi=./GraphProt_CLIP_sequences/C17ORF85_Baltz2012.train.positives.fa \
            --nega=./GraphProt_CLIP_sequences/C17ORF85_Baltz2012.train.negatives.fa \
            --model_type=DRSN--train=True --n_epochs=50')
            for j in range(10):
                a = RunCmd()
                a.cmd_run('python MARSNet.py \
                --testfile=./GraphProt_CLIP_sequences/C17ORF85_Baltz2012.ls.positives.fa \
                --nega=./GraphProt_CLIP_sequences/C17ORF85_Baltz2012.ls.negatives.fa \
                --model_type=DRSN --predict=True')


    elif b == 3:
        print(b)
        all = glob.glob("./model*")
        for h in all:
            printDRSN
            os.remove(h)

        for i in range(1,4):
            print('train：',i)
            a = RunCmd()
            a.cmd_run('python MARSNet.py \
                    --posi=./GraphProt_CLIP_sequences/C22ORF28_Baltz2012.train.positives.fa \
                    --nega=./GraphProt_CLIP_sequences/C22ORF28_Baltz2012.train.negatives.fa \
                    --model_type=DRSN --train=True --n_epochs=50')
            for j in range(10):
                a = RunCmd()
                a.cmd_run('python MARSNet.py \
                        --testfile=./GraphProt_CLIP_sequences/C22ORF28_Baltz2012.ls.positives.fa \
                        --nega=./GraphProt_CLIP_sequences/C22ORF28_Baltz2012.ls.negatives.fa \
                        --model_type=DRSN --predict=True')

    elif b == 4:
        print(b)

        all = glob.glob("./model*")
        for h in all:
            print(h)
            os.remove(h)
        for i in range(1,4):
            print('train：',i)
            a = RunCmd()
            a.cmd_run('python MARSNet.py \
                    --posi=./GraphProt_CLIP_sequences/CAPRIN1_Baltz2012.train.positives.fa \
                    --nega=./GraphProt_CLIP_sequences/CAPRIN1_Baltz2012.train.negatives.fa \
                    --model_type=DRSN --train=True --n_epochs=50')
            for j in range(10):
                a = RunCmd()
                a.cmd_run('python MARSNet.py \
                        --testfile=./GraphProt_CLIP_sequences/CAPRIN1_Baltz2012.ls.positives.fa \
                        --nega=./GraphProt_CLIP_sequences/CAPRIN1_Baltz2012.ls.negatives.fa \
                        --model_type=DRSN --predict=True')


    elif b == 5:
        print(b)
        all = glob.glob("./model*")
        for h in all:
            print(h)
            os.remove(h)
        for i in range(1,4):
            print('train：',i)
            a = RunCmd()
            a.cmd_run('python MARSNet.py \
            --posi=./GraphProt_CLIP_sequences/CLIPSEQ_AGO2.train.positives.fa \
            --nega=./GraphProt_CLIP_sequences/CLIPSEQ_AGO2.train.negatives.fa \
            --model_type=DRSN --train=True --n_epochs=50')
            for j in range(10):
                a = RunCmd()
                a.cmd_run('python MARSNet.py \
                --testfile=./GraphProt_CLIP_sequences/CLIPSEQ_AGO2.ls.positives.fa \
                --nega=./GraphProt_CLIP_sequences/CLIPSEQ_AGO2.ls.negatives.fa \
                --model_type=DRSN --predict=True')
    elif b == 6:
        print(b)
        all = glob.glob("./model*")
        for h in all:
            print(h)
            os.remove(h)
        for i in range(1,4):
            print('train：',i)
            a = RunCmd()
            a.cmd_run('python MARSNet.py \
               --posi=./GraphProt_CLIP_sequences/CLIPSEQ_ELAVL1.train.positives.fa \
               --nega=./GraphProt_CLIP_sequences/CLIPSEQ_ELAVL1.train.negatives.fa \
               --model_type=DRSN --train=True --n_epochs=50')
            for j in range(10):
                a = RunCmd()
                a.cmd_run('python MARSNet.py \
                   --testfile=./GraphProt_CLIP_sequences/CLIPSEQ_ELAVL1.ls.positives.fa \
                   --nega=./GraphProt_CLIP_sequences/CLIPSEQ_ELAVL1.ls.negatives.fa \
                   --model_type=DRSN --predict=True')
    elif b == 7:
        print(b)
        all = glob.glob("./model*")
        for h in all:
            print(h)
            os.remove(h)
        for i in range(1,4):
            print('train：',i)
            a = RunCmd()
            a.cmd_run('python MARSNet.py \
            --posi=./GraphProt_CLIP_sequences/CLIPSEQ_SFRS1.train.positives.fa \
            --nega=./GraphProt_CLIP_sequences/CLIPSEQ_SFRS1.train.negatives.fa \
            --model_type=DRSN --train=True --n_epochs=50')
            for j in range(10):
                a = RunCmd()
                a.cmd_run('python MARSNet.py \
                --testfile=./GraphProt_CLIP_sequences/CLIPSEQ_SFRS1.ls.positives.fa \
                --nega=./GraphProt_CLIP_sequences/CLIPSEQ_SFRS1.ls.negatives.fa \
                --model_type=DRSN --predict=True')

    elif b == 8:
        print(b)
        all = glob.glob("./model*")
        for h in all:
            print(h)
            os.remove(h)
        for i in range(1,4):
            print('train：',i)
            a = RunCmd()
            a.cmd_run('python MARSNet.py \
            --posi=./GraphProt_CLIP_sequences/ICLIP_HNRNPC.train.positives.fa \
            --nega=./GraphProt_CLIP_sequences/ICLIP_HNRNPC.train.negatives.fa \
            --model_type=DRSN --train=True --n_epochs=50')
            for j in range(10):
                a = RunCmd()
                a.cmd_run('python MARSNet.py \
                --testfile=./GraphProt_CLIP_sequences/ICLIP_HNRNPC.ls.positives.fa \
                --nega=./GraphProt_CLIP_sequences/ICLIP_HNRNPC.ls.negatives.fa \
                --model_type=DRSN --predict=True')

    elif b ==9:
        print(b)
        all = glob.glob("./model*")
        for h in all:
            print(h)
            os.remove(h)
        for i in range(1,4):
            print('train：',i)
            a = RunCmd()
            a.cmd_run('python MARSNet.py \
            --posi=./GraphProt_CLIP_sequences/ICLIP_TDP43.train.positives.fa \
            --nega=./GraphProt_CLIP_sequences/ICLIP_TDP43.train.negatives.fa \
            --model_type=DRSN --train=True --n_epochs=50')
            for j in range(10):
                a = RunCmd()
                a.cmd_run('python MARSNet.py \
                --testfile=./GraphProt_CLIP_sequences/ICLIP_TDP43.ls.positives.fa \
                --nega=./GraphProt_CLIP_sequences/ICLIP_TDP43.ls.negatives.fa \
                --model_type=DRSN --predict=True')
    elif b == 10:
        print(b)
        all = glob.glob("./model*")
        for h in all:
            print(h)
            os.remove(h)
        for i in range(1,4):
            print('train：',i)
            a = RunCmd()
            a.cmd_run('python MARSNet.py \
            --posi=./GraphProt_CLIP_sequences/ICLIP_TIA1.train.positives.fa \
            --nega=./GraphProt_CLIP_sequences/ICLIP_TIA1.train.negatives.fa \
            --model_type=DRSN --train=True --n_epochs=50')
            for j in range(10):
                a = RunCmd()
                a.cmd_run('python MARSNet.py \
                --testfile=./GraphProt_CLIP_sequences/ICLIP_TIA1.ls.positives.fa \
                --nega=./GraphProt_CLIP_sequences/ICLIP_TIA1.ls.negatives.fa \
                --model_type=DRSN --predict=True')
    elif b == 11:
        print(b)
        all = glob.glob("./model*")
        for h in all:
            print(h)
            os.remove(h)
        for i in range(1,4):
            print('train：',i)
            a = RunCmd()
            a.cmd_run('python MARSNet.py \
            --posi=./GraphProt_CLIP_sequences/ICLIP_TIAL1.train.positives.fa \
            --nega=./GraphProt_CLIP_sequences/ICLIP_TIAL1.train.negatives.fa \
            --model_type=DRSN --train=True --n_epochs=50')
            for j in range(10):
                a = RunCmd()
                a.cmd_run('python MARSNet.py \
                --testfile=./GraphProt_CLIP_sequences/ICLIP_TIAL1.ls.positives.fa \
                --nega=./GraphProt_CLIP_sequences/ICLIP_TIAL1.ls.negatives.fa \
                --model_type=DRSN --predict=True')
    elif b == 12:
        print(b)
        all = glob.glob("./model*")
        for h in all:
            print(h)
            os.remove(h)
        for i in range(1,4):
            print('train：',i)
            a = RunCmd()
            a.cmd_run('python MARSNet.py \
            --posi=./GraphProt_CLIP_sequences/PARCLIP_AGO1234.train.positives.fa \
            --nega=./GraphProt_CLIP_sequences/PARCLIP_AGO1234.train.negatives.fa \
            --model_type=DRSN --train=True --n_epochs=50')
            for j in range(10):
                a = RunCmd()
                a.cmd_run('python MARSNet.py \
                --testfile=./GraphProt_CLIP_sequences/PARCLIP_AGO1234.ls.positives.fa \
                --nega=./GraphProt_CLIP_sequences/PARCLIP_AGO1234.ls.negatives.fa \
                --model_type=DRSN --predict=True')
    elif b == 13:
        print(b)
        all = glob.glob("./model*")
        for h in all:
            print(h)
            os.remove(h)
        for i in range(1,4):
            print('train：',i)
            a = RunCmd()
            a.cmd_run('python MARSNet.py \
            --posi=./GraphProt_CLIP_sequences/PARCLIP_ELAVL1.train.positives.fa \
            --nega=./GraphProt_CLIP_sequences/PARCLIP_ELAVL1.train.negatives.fa \
            --model_type=DRSN --train=True --n_epochs=50')
            for j in range(10):
                a = RunCmd()
                a.cmd_run('python MARSNet.py \
                --testfile=./GraphProt_CLIP_sequences/PARCLIP_ELAVL1.ls.positives.fa \
                --nega=./GraphProt_CLIP_sequences/PARCLIP_ELAVL1.ls.negatives.fa \
                --model_type=DRSN --predict=True')
    elif b == 14:
        print(b)
        all = glob.glob("./model*")
        for h in all:
            print(h)
            os.remove(h)
        for i in range(1,4):
            print('train：',i)
            a = RunCmd()
            a.cmd_run('python MARSNet.py \
            --posi=./GraphProt_CLIP_sequences/PARCLIP_ELAVL1A.train.positives.fa \
            --nega=./GraphProt_CLIP_sequences/PARCLIP_ELAVL1A.train.negatives.fa \
            --model_type=DRSN --train=True --n_epochs=50')
            for j in range(10):
                a = RunCmd()
                a.cmd_run('python MARSNet.py \
                --testfile=./GraphProt_CLIP_sequences/PARCLIP_ELAVL1A.ls.positives.fa \
                --nega=./GraphProt_CLIP_sequences/PARCLIP_ELAVL1A.ls.negatives.fa \
                --model_type=DRSN --predict=True')
    elif b == 15:
        print(b)
        all = glob.glob("./model*")
        for h in all:
            print(h)
            os.remove(h)
        for i in range(1,4):
            print('train：',i)
            a = RunCmd()
            a.cmd_run('python MARSNet.py \
            --posi=./GraphProt_CLIP_sequences/PARCLIP_EWSR1.train.positives.fa \
            --nega=./GraphProt_CLIP_sequences/PARCLIP_EWSR1.train.negatives.fa \
            --model_type=DRSN --train=True --n_epochs=50')
            for j in range(10):
                a = RunCmd()
                a.cmd_run('python MARSNet.py \
                --testfile=./GraphProt_CLIP_sequences/PARCLIP_EWSR1.ls.positives.fa \
                --nega=./GraphProt_CLIP_sequences/PARCLIP_EWSR1.ls.negatives.fa \
                --model_type=DRSN --predict=True')
    elif b == 16:
        print(b)
        all = glob.glob("./model*")
        for h in all:
            print(h)
            os.remove(h)
        for i in range(1,4):
            print('train：',i)
            a = RunCmd()
            a.cmd_run('python MARSNet.py \
            --posi=./GraphProt_CLIP_sequences/PARCLIP_FUS.train.positives.fa \
            --nega=./GraphProt_CLIP_sequences/PARCLIP_FUS.train.negatives.fa \
            --model_type=DRSN --train=True --n_epochs=50')
            for j in range(10):
                a = RunCmd()
                a.cmd_run('python MARSNet.py \
                --testfile=./GraphProt_CLIP_sequences/PARCLIP_FUS.ls.positives.fa \
                --nega=./GraphProt_CLIP_sequences/PARCLIP_FUS.ls.negatives.fa \
                --model_type=DRSN --predict=True')
    elif b == 17:
        print(b)
        all = glob.glob("./model*")
        for h in all:
            print(h)
            os.remove(h)
        for i in range(1,4):
            print('train：',i)
            a = RunCmd()
            a.cmd_run('python MARSNet.py \
               --posi=./GraphProt_CLIP_sequences/PARCLIP_HUR.train.positives.fa \
               --nega=./GraphProt_CLIP_sequences/PARCLIP_HUR.train.negatives.fa \
               --model_type=DRSN --train=True --n_epochs=50')
            for j in range(10):
                a = RunCmd()
                a.cmd_run('python MARSNet.py \
                   --testfile=./GraphProt_CLIP_sequences/PARCLIP_HUR.ls.positives.fa \
                   --nega=./GraphProt_CLIP_sequences/PARCLIP_HUR.ls.negatives.fa \
                   --model_type=DRSN --predict=True')

    elif b == 18:
        print(b)
        all = glob.glob("./model*")
        for h in all:
            print(h)
            os.remove(h)
        for i in range(1,4):
            print('train：',i)
            a = RunCmd()
            a.cmd_run('python MARSNet.py \
            --posi=./GraphProt_CLIP_sequences/PARCLIP_IGF2BP123.train.positives.fa \
            --nega=./GraphProt_CLIP_sequences/PARCLIP_IGF2BP123.train.negatives.fa \
            --model_type=DRSN --train=True --n_epochs=50')
            for j in range(10):
                a = RunCmd()
                a.cmd_run('python MARSNet.py \
                --testfile=./GraphProt_CLIP_sequences/PARCLIP_IGF2BP123.ls.positives.fa \
                --nega=./GraphProt_CLIP_sequences/PARCLIP_IGF2BP123.ls.negatives.fa \
                --model_type=DRSN --predict=True')
    elif b == 19:
        print(b)
        all = glob.glob("./model*")
        for h in all:
            print(h)
            os.remove(h)
        for i in range(1,4):
            print('train：',i)
            a = RunCmd()
            a.cmd_run('python MARSNet.py \
            --posi=./GraphProt_CLIP_sequences/PARCLIP_MOV10_Sievers.train.positives.fa \
            --nega=./GraphProt_CLIP_sequences/PARCLIP_MOV10_Sievers.train.negatives.fa \
            --model_type=DRSN --train=True --n_epochs=50')
            for j in range(10):
                a = RunCmd()
                a.cmd_run('python MARSNet.py \
                --testfile=./GraphProt_CLIP_sequences/PARCLIP_MOV10_Sievers.ls.positives.fa \
                --nega=./GraphProt_CLIP_sequences/PARCLIP_MOV10_Sievers.ls.negatives.fa \
                --model_type=DRSN --predict=True')
    elif b == 20:
        print(b)
        all = glob.glob("./model*")
        for h in all:
            print(h)
            os.remove(h)
        for i in range(1,4):
            print('train：',i)
            a = RunCmd()
            a.cmd_run('python MARSNet.py \
            --posi=./GraphProt_CLIP_sequences/PARCLIP_PUM2.train.positives.fa \
            --nega=./GraphProt_CLIP_sequences/PARCLIP_PUM2.train.negatives.fa \
            --model_type=DRSN --train=True --n_epochs=50')
            for j in range(10):
                a = RunCmd()
                a.cmd_run('python MARSNet.py \
                --testfile=./GraphProt_CLIP_sequences/PARCLIP_PUM2.ls.positives.fa \
                --nega=./GraphProt_CLIP_sequences/PARCLIP_PUM2.ls.negatives.fa \
                --model_type=DRSN --predict=True')
    elif b == 21:
        print(b)
        all = glob.glob("./model*")
        for h in all:
            print(h)
            os.remove(h)
        for i in range(1,4):
            print('train：',i)
            a = RunCmd()
            a.cmd_run('python MARSNet.py \
            --posi=./GraphProt_CLIP_sequences/PARCLIP_QKI.train.positives.fa \
            --nega=./GraphProt_CLIP_sequences/PARCLIP_QKI.train.negatives.fa \
            --model_type=DRSN --train=True --n_epochs=50')
            for j in range(10):
                a = RunCmd()
                a.cmd_run('python MARSNet.py \
                --testfile=./GraphProt_CLIP_sequences/PARCLIP_QKI.ls.positives.fa \
                --nega=./GraphProt_CLIP_sequences/PARCLIP_QKI.ls.negatives.fa \
                --model_type=DRSN --predict=True')

    elif b == 22:
        print(b)
        all = glob.glob("./model*")
        for h in all:
            print(h)
            os.remove(h)
        for i in range(1,4):
            print('train：',i)
            a = RunCmd()
            a.cmd_run('python MARSNet.py \
            --posi=./GraphProt_CLIP_sequences/PARCLIP_TAF15.train.positives.fa \
            --nega=./GraphProt_CLIP_sequences/PARCLIP_TAF15.train.negatives.fa \
            --model_type=DRSN --train=True --n_epochs=50')
            for j in range(10):
                a = RunCmd()
                a.cmd_run('python MARSNet.py \
                --testfile=./GraphProt_CLIP_sequences/PARCLIP_TAF15.ls.positives.fa \
                --nega=./GraphProt_CLIP_sequences/PARCLIP_TAF15.ls.negatives.fa \
                --model_type=DRSN --predict=True')

    elif b == 23:
        print(b)
        all = glob.glob("./model*")
        for h in all:
            print(h)
            os.remove(h)
        for i in range(1,4):
            print('train：',i)
            a = RunCmd()
            a.cmd_run('python MARSNet.py \
            --posi=./GraphProt_CLIP_sequences/PTBv1.train.positives.fa \
            --nega=./GraphProt_CLIP_sequences/PTBv1.train.negatives.fa \
            --model_type=DRSN --train=True --n_epochs=50')
            for j in range(10):
                a = RunCmd()
                a.cmd_run('python MARSNet.py \
                --testfile=./GraphProt_CLIP_sequences/PTBv1.ls.positives.fa \
                --nega=./GraphProt_CLIP_sequences/PTBv1.ls.negatives.fa \
                --model_type=DRSN --predict=True')


    elif b == 24:
        print(b)
        all = glob.glob("./model*")
        for h in all:
            print(h)
            os.remove(h)
        for i in range(1,4):
            print('train：',i)
            a = RunCmd()
            a.cmd_run('python MARSNet.py \
            --posi=./GraphProt_CLIP_sequences/ZC3H7B_Baltz2012.train.positives.fa \
            --nega=./GraphProt_CLIP_sequences/ZC3H7B_Baltz2012.train.negatives.fa \
            --model_type=DRSN --train=True --n_epochs=50')
            for j in range(10):
                a = RunCmd()
                a.cmd_run('python MARSNet.py \
                --testfile=./GraphProt_CLIP_sequences/ZC3H7B_Baltz2012.ls.positives.fa \
                --nega=./GraphProt_CLIP_sequences/ZC3H7B_Baltz2012.ls.negatives.fa \
                --model_type=DRSN --predict=True')

