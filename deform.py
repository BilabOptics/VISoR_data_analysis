import sys, cv2, os
import numpy as np
from os.path import join
from subprocess import call
from shutil import copyfile
import SimpleITK as sitk
from PIL import Image

# Download from http://elastix.isi.uu.nl/
ELASTIX_EXEC = "C:/Users/chaoyu/Downloads/elastix_windows64_v4.8/elastix"
TRANSFORMIX_EXEC = "C:/Users/chaoyu/Downloads/elastix_windows64_v4.8/transformix"
cfgpath = os.path.dirname(__file__)
TEMPLETE = join(cfgpath, "template.tif")

START_SLICE = 3
PICTURE_SIZE = (600, 670)
GRID_SIZE = (2, 78, 87)
flip = False

def generateChessboard(dstfile, shape, dtype=np.uint8):
    cb = np.zeros(shape, dtype)
    for i in range(0, shape[0], 16):
        for j in range(0, shape[1], 16):
            if (i + j)%32 == 0:
                cb[i:min(i + 16, shape[0]), j:min(j + 16, shape[1])] += 255
    cv2.imwrite(dstfile, cb)


def elastix(fix, mov, out, p, t0 = None):
    param = [ELASTIX_EXEC, "-f", fix, "-m", mov, "-out", out]
    for par in p:
        param.append("-p")
        param.append(par)
    if not t0 is None:
        param.append("-t0")
        param.append(t0)
    if not call(param) == 0:
        raise Exception("Elastix error")


def transformix(inp, out, tp, def_ = None):
    param = []
    if not def_ is None:
        param = [TRANSFORMIX_EXEC, "-def", def_, "-out", out, "-tp", tp]
    else:
        param = [TRANSFORMIX_EXEC, "-in", inp, "-out", out, "-tp", tp]
    if not call(param) == 0:
        raise Exception("Elastix error")


def preProcess(srcfile, dstfile, shape):
    src = cv2.imread(srcfile, -1)
    #src = np.uint8(src)
    if src.dtype == np.uint16:
        src = np.uint8(np.clip((cv2.log(np.float32(src)) - 4.6) * 39.4, 0, 255))
    #src = np.uint16(cv2.exp(np.float32(src) / 39.4 + 4.6)) - 145
    img = np.zeros(shape, src.dtype)
    roi = img[int((shape[0] - src.shape[0]) / 2): int((shape[0] + src.shape[0]) / 2),
          int((shape[1] - src.shape[1]) / 2): int((shape[1] + src.shape[1]) / 2)]
    np.copyto(roi, src)
    #_, img = cv2.threshold(255 - img, 240, 0, cv2.THRESH_TRUNC)
    #img = cv2.medianBlur(255 - img, 3)
    #img = 255 - img
    msk = np.zeros((shape[0] + 2, shape[1] + 2), np.uint8)
    cv2.floodFill(img, msk, (0, 0), 20, 0, 20, cv2.FLOODFILL_FIXED_RANGE)
    img = np.uint16(img)
    #cv2.normalize(img, img, 16384, cv2.NORM_L2)
    cv2.imwrite(dstfile, img)


def convertRawImage(srcfile, dstfile, shape):
    img = np.fromfile(srcfile, np.int16)
    img = np.reshape(img, shape)
    cv2.imwrite(dstfile, img)


def editParamFile(srcfile, dstfile, initialtp = None, a = 1.0):
    src = open(srcfile, 'r')
    dst = open(dstfile, 'w')
    while 1:
        line = src.readline()
        if len(line) == 0:
            break
        l = line.split(' ')
        if l[0] == "(InitialTransformParametersFileName":
            if not initialtp is None:
                line = l[0] + " " + initialtp + ")\n"
        if l[0] == "(TransformParameters":
            for p in l:
                if p[0] == '(':
                    line = p + " "
                elif p[len(p) - 2] == ')':
                    line += "{0:.6f}".format(float(p[0:len(p) - 2]) * a) + ")\n"
                else:
                    line += "{0:.6f} ".format(float(p) * a)
        dst.write(line)


def addParamFile(srcfile1, srcfile2, dstfile, a = 1.0, b = 1.0):
    f1 = sitk.ReadParameterFile(srcfile1)
    f2 = sitk.ReadParameterFile(srcfile2)
    replace_list = ['TransformParameters']
    p = None
    for n in replace_list:
        p1 = np.array([float(f) for f in f1[n]])
        s1 = (2, int(f1['GridSize'][1]), int(f1['GridSize'][0]))
        p1 = p1.reshape(s1)
        s1 = (2, min(s1[1], GRID_SIZE[1]), min(s1[2], GRID_SIZE[2]))
        p1_ = np.zeros(GRID_SIZE, p1.dtype)
        np.copyto(p1_[0:s1[0], 0:s1[1], 0:s1[2]], p1[0:s1[0], 0:s1[1], 0:s1[2]])
        p2 = np.array([float(f) for f in f2[n]])
        s2 = (2, int(f2['GridSize'][1]), int(f2['GridSize'][0]))
        p2 = p2.reshape(s2)
        s2 = (2, min(s2[1], GRID_SIZE[1]), min(s2[2], GRID_SIZE[2]))
        p2_ = np.zeros(GRID_SIZE, p2.dtype)
        np.copyto(p2_[0:s2[0], 0:s2[1], 0:s2[2]], p2[0:s2[0], 0:s2[1], 0:s2[2]])
        p = p1_ * a + p2_ * b
        print(p1)
        print(p2)
        print(s1)
        print(s2)
        p = np.reshape(p, p.size)
        print(p)
        print(p.shape)
        f1[n] = ['{0:.6f}'.format(float(f)) for f in p]
        print(f1[n])
    print(f1['NumberOfParameters'])
    f1['NumberOfParameters'] = [str(int(p.size))]
    f1['GridSize'] = [str(int(GRID_SIZE[2])), str(int(GRID_SIZE[1]))]
    replace_list = ['GridSpacing', 'GridOrigin']
    for n in replace_list:
        p1 = np.array([float(f) for f in f1[n]])
        p2 = np.array([float(f) for f in f2[n]])
        p = p1 * a / (a + b) + p2 * b / (a + b)
        f1[n] = ['{0:.6f}'.format(f) for f in p]
    sitk.WriteParameterFile(f1, dstfile)


def deform(srcdirlist, dstdir, tmpdir):
    #'''
    grid = join(cfgpath, "cb.tif")
    generateChessboard(grid, PICTURE_SIZE)
    sl3 = None
    for f in srcdirlist:
        if f[0] == START_SLICE:
            sl3 = f[1]
            break
    if sl3 is None:
        return
    step1 = join(tmpdir, "step1")
    if not os.path.exists(step1):
        os.mkdir(step1)
    e = 0
    for f in srcdirlist:
        imglist = os.listdir(f[1])
        e = len(imglist) - 1
        for i in range(0, len(imglist)):
            preProcess(join(f[1], imglist[i]), join(step1, str(f[0]) + "_" + imglist[i]), PICTURE_SIZE)
    step2 = join(tmpdir, "step2")
    img2 = join(step2, "img")
    par1 = join(step2, "par")
    par = join(dstdir, "par")
    if not os.path.exists(step2):
        os.mkdir(step2)
    if not os.path.exists(img2):
        os.mkdir(img2)
    if not os.path.exists(par1):
        os.mkdir(par1)
    if not os.path.exists(par):
        os.mkdir(par)

    if flip:
        j_range = range(e - 1, 0, -1)
        s = e
    else:
        j_range = range(0, e + 1, 1)
        s = 0
    ct = 1
    preProcess(join(cfgpath, "tp_4.tif"), join(step1, "tp_4.tif"), PICTURE_SIZE)
    elastix(join(step1, "tp_4.tif"),
            join(step1, str(START_SLICE) + "_" + str(s) + ".tif"),
            step1,
            [join(cfgpath, "parameters_Rigid.txt")])
    convertRawImage(join(step1, "result.0.raw"), join(img2, str(ct) + ".tif"), PICTURE_SIZE)
    copyfile(join(step1, "TransformParameters.0.txt"),
             join(par, "r.p." + str(START_SLICE) + ".txt"))
    elastix(grid, grid, step1, [join(cfgpath, "parameters_Inverse_Rigid.txt")],
            join(step1, "TransformParameters.0.txt"))
    editParamFile(join(step1, "TransformParameters.0.txt"),
                  join(par, "r.i." + str(START_SLICE) + ".txt"), "\"NoInitialTransform\"")
    i = START_SLICE + 1
    while os.path.exists(join(step1, str(i) + "_" + str(0) + ".tif")):
        transformix(join(step1, str(i - 1) + "_" + str(e - s) + ".tif"),
                    step1,
                    join(par, "r.p." + str(i - 1) + ".txt"))
        convertRawImage(join(step1, "result.raw"), join(img2, str(ct) + ".tif"), PICTURE_SIZE)
        elastix(join(img2, str(ct) + ".tif"),
                join(step1, str(i) + "_" + str(s) + ".tif"),
                step1,
                [join(cfgpath, "parameters_Rigid.txt"),
                 join(cfgpath, "parameters_BSpline.txt")])
        copyfile(join(step1, "TransformParameters.0.txt"),
                 join(par, "r.p." + str(i) + ".txt"))
        editParamFile(join(step1, "TransformParameters.1.txt"),
                      join(par1, "b.p." + str(i) + ".txt"),
                      join(par, "r.p." + str(i) + ".txt"), 0.5)
        elastix(grid,
                grid,
                step1, [join(cfgpath, "parameters_Inverse_Rigid.txt")],
                join(step1, "TransformParameters.0.txt"))
        editParamFile(join(step1, "TransformParameters.0.txt"),
                      join(par, "r.i." + str(i) + ".txt"), "\"NoInitialTransform\"")
        editParamFile(join(step1, "TransformParameters.1.txt"),
                      join(step1, "t.txt"), "\"NoInitialTransform\"", 0.5)
        elastix(grid,
                grid,
                step1, [join(cfgpath, "parameters_Inverse_bs.txt")],
                join(step1, "t.txt"))
        copyfile(join(step1, "TransformParameters.0.txt"),
                 join(par1, "b.i." + str(i) + ".txt"))
        editParamFile(join(step1, "TransformParameters.0.txt"),
                      join(par, "b.i." + str(i) + "." + str(s) + ".txt"),
                      join(par, "r.i." + str(i) + ".txt"))
        for j in j_range:
            if os.path.exists(join(par1, "b.p." + str(i - 1) + ".txt")):
                addParamFile(join(par1, "b.p." + str(i - 1) + ".txt"),
                             join(par1, "b.i." + str(i) + ".txt"),
                             join(par, "b.p." + str(i - 1) + "." + str(j) + ".txt"),
                             abs(float(e - s - j)) / float(e) * 1,
                             abs(float(s + j)) / float(e) * 1)
            else:
                editParamFile(join(par1, "b.i." + str(i) + ".txt"),
                              join(par, "b.p." + str(i - 1) + "." + str(j) + ".txt"),
                              join(par, "r.p." + str(i - 1) + ".txt"), abs(float(s + j)) / float(e) * 1)
            transformix(join(step1, str(i - 1) + "_" + str(j) + ".tif"),
                        step1,
                        join(par, "b.p." + str(i - 1) + "." + str(j) + ".txt"))
            convertRawImage(join(step1, "result.raw"), join(img2, str(ct) + ".tif"), PICTURE_SIZE)
            ct += 1
        i += 1

    for j in j_range:
        editParamFile(join(par1, "b.p." + str(i - 1) + ".txt"),
                      join(par, "b.p." + str(i - 1) + "." + str(j) + ".txt"),
                      join(par, "r.p." + str(i - 1) + ".txt"), abs(float(e - s - j)) / float(e) * 1)
        transformix(join(step1, str(i - 1) + "_" + str(j) + ".tif"),
                    step1,
                    join(par, "b.p." + str(i - 1) + "." + str(j) + ".txt"))
        convertRawImage(join(step1, "result.raw"), join(img2, str(ct) + ".tif"), PICTURE_SIZE)
        ct += 1
    #'''
    step2 = join(tmpdir, "step2")
    img2 = join(step2, "img")
    ct = 1
    stack = []
    while 1:
        img = join(img2, str(ct) + ".tif")
        if not os.path.exists(img):
            break
        stack.append(img)
        ct += 1
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(stack)
    stack = reader.Execute()
    #stack = sitk.Median(stack)
    sitk.WriteImage(stack, join(step2, "brain.tif"))

    elastix(join(cfgpath, "template.tif"), join(step2, "brain.tif"), step2,
            [join(cfgpath, "parameters_Rigid.txt"),
             join(cfgpath, "parameters_Affine.txt"),
             join(cfgpath, "parameters_BSpline_3d.txt")])
    copyfile(join(step2, "TransformParameters.0.txt"),
             join(dstdir, "br.r.p.txt"))
    editParamFile(join(step2, "TransformParameters.1.txt"),
                  join(dstdir, "br.a.p.txt"),
                  join(dstdir, "br.r.p.txt"))
    editParamFile(join(step2, "TransformParameters.2.txt"),
                  join(dstdir, "br.b.p.txt"),
                  join(dstdir, "br.a.p.txt"))
    elastix(join(step2, "brain.tif"), TEMPLETE, step2,
            [join(cfgpath, "parameters_Inverse_bs_3d.txt")],
            join(dstdir, "br.b.p.txt"))
    editParamFile(join(step2, "TransformParameters.0.txt"),
                  join(dstdir, "br.b.i.txt"),
                  "\"NoInitialTransform\"")

if __name__ == "__main__":
    dir = "/run/media/o/data_ycy/testdata/cfos_1481_full/test"
    dirlist = [f
               for f in os.listdir(dir)
               if os.path.isdir(join(dir, f))]
    lst = []
    for f in dirlist:
        lst.append((int(f), os.path.join(dir, f)))
    lst.sort(cmp=lambda x, y: x[0] - y[0])
    deform(lst,
           "/run/media/o/data_ycy/testdata/cfos_1481_full/n_res",
           "/run/media/o/data_ycy/testdata/cfos_1481_full/n_tmp")

