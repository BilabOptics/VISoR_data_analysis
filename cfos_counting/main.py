import sys
from os.path import join
from deform import *
from registration import *
from subprocess import call
from classify import *
import flatten_fix


#FR_EXEC = "D:\\Hao\\reconstructed\\cfos-FS5-2406_488"
#brain_name = "cfos-FS2-461_488"


def checkdir(srcdir):
    return [f
            for f in os.listdir(srcdir)
            if os.path.isdir(join(srcdir, f + "/origin"))]


def main(srcdir):
    cfgpath = os.path.dirname(__file__)
    #srcdir = 'D:\\Hao\\reconstructed\\cfos-FS2-280_488'
    #dstdir = 'D:/Hao/reconstructed'
    #datadir = checkdir(srcdir)
    #for d in datadir:
    #    dc = join(srcdir, d + "/compressed")
    #    if not os.path.exists(dc):
    #        os.mkdir(dc)
        #compress(join(srcdir, d + "/origin"), dc)
    #call([FR_EXEC, "-s", srcdir, join(dstdir, BRAIN_NAME), "c"])
    #if not os.path.exists(srcdir):
    #    os.mkdir(srcdir)
    tmpdir = join(srcdir, "tmp")
    resdir = join(srcdir, "res")
    datadir = join(srcdir, os.path.split(srcdir)[1])
    if not os.path.exists(tmpdir):
        os.mkdir(tmpdir)
    if not os.path.exists(resdir):
        os.mkdir(resdir)
    r1 = [f for f in os.listdir(datadir)
          if os.path.isdir(join(datadir, f))]
    r2 = []
    r3 = []
    for f in r1:
        r2.append((int(f.split('-')[0]), join(datadir, f, "25")))
        r3.append((int(f.split('-')[0]), join(datadir, f)))
    r2.sort(key=lambda x: x[0])
    r3.sort(key=lambda x: x[0])
    #for f in r3:
    #    flatten_fix.flatten_fix(f[1])
    #model, sc = train(join(cfgpath, "D:/chaoyu/cfos/train.csv"))
    #for f in r3:
    #    classify(model, join(f[1], "cells.csv"), join(f[1], "classified.csv"), sc)
    deform(r2, resdir, tmpdir)
    registrate(r3, resdir, tmpdir)


if __name__ == "__main__":
    src_list = [os.path.join('D:/Hao/reconstructed/cfos_counting', f_)
                for f_ in os.listdir('D:/Hao/reconstructed/cfos_counting')]
    for d in src_list[12:]:
        print(d)
        main(d)
