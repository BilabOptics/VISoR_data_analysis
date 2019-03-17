from classify import *
from deform import *
from allensdk.api.queries.ontologies_api import OntologiesApi
from allensdk.core.structure_tree import StructureTree
import pickle

PIXEL_SIZE = 0.494
SRC_PIXEL_SIZE = 0.494 * 8
MAP_PIXEL_SIZE = 25

n_slice = 300 / MAP_PIXEL_SIZE + 1

def readTransfomixOutputPoints(srcfile):
    points = []
    file = open(srcfile)
    while 1:
        l = file.readline()
        if len(l) < 10:
            break
        x = float(l.split(';')[4].split(' ')[4])
        y = float(l.split(';')[4].split(' ')[5])
        if not l.split(';')[4].split(' ')[6].split('.')[0].split('-')[-1].isdigit():
            points.append((x, y))
        else:
            z = float(l.split(';')[4].split(' ')[6])
            points.append((x, y, z))
    return points


def writeTransfomixInputPoints(dstfile, points, ndim):
    df = open(dstfile, "w")
    df.write("point\n")
    df.write(str(len(points)) + "\n")
    for c in points:
        df.write(str(c[0]))
        for i in range(1, ndim):
            df.write(" {0}".format(str(c[i])))
        df.write("\n")
    df.close()


def getOriginalPos(registratedPos, brainPath):
    tmpfile = join(brainPath, "tmp/t.txt")
    writeTransfomixInputPoints(tmpfile, registratedPos, 3)
    transformix(None, join(brainPath, "res"), join(brainPath, "res/br.b.i.txt"), tmpfile)
    posf = readTransfomixOutputPoints(join(brainPath, "res/outputpoints.txt"))
    slices = []
    posb = []
    abo = []
    und = []

    for f_ in os.listdir(join(brainPath, os.path.dirname(brainPath))):
        if int(f_.split('-')[0]) < START_SLICE or not \
                os.path.exists(join(brainPath, "res/par/r.i.") + str(f_.split('-')[0]) + ".txt"):
            continue
        slices.append(f_)
        for i in range(int(n_slice)):
            posb.append([])
    slices.sort(cmp=lambda x, y: int(x.split(',')[0]) - int(y.split(',')[0]))
    for f in slices:
        abo.append(cv2.imread(join(join(brainPath, os.path.dirname(brainPath)), f + "/a.tif"), -1))
        und.append(cv2.imread(join(join(brainPath, os.path.dirname(brainPath)), f + "/u.tif"), -1))

    for i in range(len(posf)):
        if 0 <= int(posf[i][2]) < len(posb):
            posb[int(posf[i][2])] = (i, posf[i])

    posr = {}
    for i in range(len(posb)):
        if len(posb[i]) == 0:
            continue
        p_ = []
        k_ = []
        for v in posb[i]:
            p_.append(v[1])
            k_.append(v[0])
        idx = int(i / n_slice) + START_SLICE
        writeTransfomixInputPoints(tmpfile, p_, 2)
        transformix(None,
                    join(brainPath, "res"),
                    join(brainPath, "res/b.p." + str(idx) + "." + str(i % n_slice) +".txt"),
                    tmpfile)
        pp = readTransfomixOutputPoints(join(brainPath, "res/outputpoints.txt"))
        for j in range(len(pp)):
            p = pp[j]
            x = p[0] / SRC_PIXEL_SIZE * MAP_PIXEL_SIZE
            y = p[1] / SRC_PIXEL_SIZE * MAP_PIXEL_SIZE
            a = abo[int(i / n_slice)][int(y)][int(x)]
            u = und[int(i / n_slice)][int(y)][int(x)]
            if flip:
                z = (300 / MAP_PIXEL_SIZE - float(i % n_slice)) * (u - a) * SRC_PIXEL_SIZE * MAP_PIXEL_SIZE / 300
            else:
                z = float(i % n_slice) * (u - a) * SRC_PIXEL_SIZE * MAP_PIXEL_SIZE / 300
            posr[k_[j]] = (idx, x, y, z)
    return posr.values()


def registrate(srcdirlist, dstdir, tmpdir):
    cfgpath = os.path.dirname(__file__)
    brmap = sitk.ReadImage(join(cfgpath, "annotation_25.nrrd"))
    brmap = sitk.GetArrayFromImage(brmap)
    brmap = np.transpose(brmap, [2, 1, 0])
    #cv2.imreadmulti(join(cfgpath, "map.tif"), brmap)
    #oapi = OntologiesApi()
    #structure_graph = oapi.get_structures_with_sets([1])
    #structure_graph = StructureTree.clean_structures(structure_graph)
    #tree = StructureTree(structure_graph)
    #with open('ontology.pkl', 'wb') as reg:
    #    pickle.dump(tree, reg)
    with open('ontology.pkl', 'rb') as reg:
        tree = pickle.load(reg)

    #regions = {0:[[-1, 'none', 'none'], [], [0, 0, 0]]}

    cellpos = []
    tmpfile = join(tmpdir, "t.txt")
    for f_ in srcdirlist:
        if f_[0] < START_SLICE or not os.path.exists(join(dstdir, "par/r.i.") + str(f_[0]) + ".txt"):
            continue
        fd = f_[1]
        idx = f_[0] - START_SLICE
        abo = cv2.imread(join(fd, "a.tif"), -1)
        und = cv2.imread(join(fd, "u.tif"), -1)
        head, cells, qf, qm = refineData(join(fd, "classified.csv"))
        x0 = float(head[2])
        y0 = float(head[3])
        cellpos0 = []
        for c in cells:
            x = (float(c[0]) - x0) / SRC_PIXEL_SIZE
            y = (float(c[1]) - y0) / SRC_PIXEL_SIZE
            if not (x < abo.shape[1] and y < abo.shape[0]):
                continue
            zz = float(c[2]) / SRC_PIXEL_SIZE
            a = abo[int(y), int(x)]
            u = und[int(y), int(x)]
            x += (PICTURE_SIZE[1] / SRC_PIXEL_SIZE * MAP_PIXEL_SIZE - abo.shape[1]) / 2
            y += (PICTURE_SIZE[0] / SRC_PIXEL_SIZE * MAP_PIXEL_SIZE - abo.shape[0]) / 2
            x *= SRC_PIXEL_SIZE / MAP_PIXEL_SIZE
            y *= SRC_PIXEL_SIZE / MAP_PIXEL_SIZE
            z = (zz - a) * 300 / MAP_PIXEL_SIZE / (u - a)
            if flip:
                z = 300 / MAP_PIXEL_SIZE - z
            if 0 / MAP_PIXEL_SIZE < z < 300 / MAP_PIXEL_SIZE and c[len(c) - 1] == "1":
                cellpos0.append((x, y, (z - 25 / MAP_PIXEL_SIZE) * 350 / 300, c))
        writeTransfomixInputPoints(tmpfile, cellpos0, 2)
        transformix(None, dstdir, join(dstdir, "par/r.i.") + str(f_[0]) + ".txt", tmpfile)
        cellposr = readTransfomixOutputPoints(join(dstdir, "outputpoints.txt"))
        if os.path.exists(join(dstdir, "par/b.p.") + str(f_[0] - 1) + ".0.txt"):
            editParamFile(join(dstdir, "par/b.p.") + str(f_[0] - 1) + ".0.txt",
                          join(dstdir, "tp.txt"),
                          join(dstdir, "par/r.i.") + str(f_[0]) + ".txt")
            transformix(None, dstdir, join(dstdir, "tp.txt"), tmpfile)
            cellposb2 = readTransfomixOutputPoints(join(dstdir, "outputpoints.txt"))
        else:
            cellposb2 = cellposr
        if os.path.exists(join(dstdir, "par/b.i.") + str(f_[0]) + ".0.txt"):
            transformix(None, dstdir, join(dstdir, "par/b.i.") + str(f_[0]) + ".0.txt", tmpfile)
            cellposb1 = readTransfomixOutputPoints(join(dstdir, "outputpoints.txt"))
        else:
            cellposb1 = cellposr
        for ct in range(len(cellposb2)):
            cr = cellposb2[ct]
            cb = cellposb1[ct]
            z = cellpos0[ct][2]
            x = (cb[0] * z + cr[0] * (300 / MAP_PIXEL_SIZE - z)) / 300 * MAP_PIXEL_SIZE
            y = (cb[1] * z + cr[1] * (300 / MAP_PIXEL_SIZE - z)) / 300 * MAP_PIXEL_SIZE
            z += idx * (300 / MAP_PIXEL_SIZE + 1)
            if x < PICTURE_SIZE[1] and y < PICTURE_SIZE[0]:
                cellpos.append((x, y, z, cellpos0[ct][3]))
    writeTransfomixInputPoints(tmpfile, cellpos, 3)
    transformix(None, dstdir, join(dstdir, "br.b.i.txt"), tmpfile)
    cellposf = readTransfomixOutputPoints(join(dstdir, "outputpoints.txt"))

    dst = open(join(dstdir, "summary_l.csv"), "w")
    summary = csv.writer(dst)
    cm = np.zeros((brmap.shape[0], brmap.shape[1], brmap.shape[2], 3), np.uint8)
    regions = {0:[[-1, 'none', 'none'], [], [0, 0, 0]]}
    for ct in range(len(cellposf)):
        cell = cellposf[ct]
        x = int(cell[0])
        y = int(cell[1])
        z = int(cell[2])
        if 0 <= x < brmap.shape[2] / 2 and 0 <= y < brmap.shape[1] and 0 <= z < brmap.shape[0]:
            label = int(brmap[z, y, x])
            try:
                regions[label][1].append(float(cellpos[ct][3][10]))
                for j in range(3):
                    cm[z, y, x, j] = regions[label][2][j]
            except KeyError:
                region = tree.get_structures_by_id([label])[0]
                regions[label] = [[region['graph_order'], region['name'], region['acronym']],
                                  [float(cellpos[ct][3][10])],
                                  region['rgb_triplet']]
            except:
                print(label)

    rows = []
    for l in regions.values():
        hist, _0 = np.histogram(l[1], bins=np.linspace(1, 5, 41))
        row = [l[0][0], l[0][1], l[0][2]]
        for h in hist:
            row.append(str(h))
        row.append(len(l[1]))
        rows.append(row)
    rows.sort(key=lambda x: int(x[0]))
    for row in rows:
        summary.writerow(row)

    dst = open(join(dstdir, "summary_r.csv"), "w")
    summary = csv.writer(dst)
    regions = {0: [[-1, 'none', 'none'], [], [0, 0, 0]]}
    #cm = np.zeros((brmap.shape[0], brmap.shape[1], brmap.shape[2], 3), np.uint8)
    for ct in range(len(cellposf)):
        cell = cellposf[ct]
        x = int(cell[0])
        y = int(cell[1])
        z = int(cell[2])
        if brmap.shape[2] / 2 <= x < brmap.shape[2] and 0 <= y < brmap.shape[1] and 0 <= z < brmap.shape[0]:
            label = int(brmap[z, y, x])
            try:
                regions[label][1].append(float(cellpos[ct][3][10]))
                for j in range(3):
                    cm[z, y, x, j] = regions[label][2][j]
            except KeyError:
                region = tree.get_structures_by_id([label])[0]
                regions[label] = [[region['graph_order'], region['name'], region['acronym']],
                                  [float(cellpos[ct][3][10])],
                                  region['rgb_triplet']]
            except:
                print(label)

    rows = []
    for l in regions.values():
        hist, _0 = np.histogram(l[1], bins=np.linspace(1, 5, 41))
        row = [l[0][0], l[0][1], l[0][2]]
        for h in hist:
            row.append(str(h))
        row.append(len(l[1]))
        rows.append(row)
    rows.sort(key=lambda x: int(x[0]))
    for row in rows:
        summary.writerow(row)

    cm = sitk.GetImageFromArray(cm)
    sitk.WriteImage(cm, join(dstdir, "cm.tif"))



if __name__ == "__main__":
    r1 = [f for f in os.listdir("/run/media/o/data_ycy/testdata/cfos-TH/cfos-TH/")
          if os.path.isdir(join("/run/media/o/data_ycy/testdata/cfos-TH/cfos-TH/", f))]
    r1.sort(cmp=lambda x,y: int(x.split('-')[0]) - int(y.split('-')[0]))
    r2 = []
    for f in r1:
        r2.append((int(f.split('-')[0]), join("/run/media/o/data_ycy/testdata/cfos-TH/cfos-TH/", f)))
    registrate(r2, "/run/media/o/data_ycy/testdata/cfos-TH/res/",
                 "/run/media/o/data_ycy/testdata/cfos-TH/tmp/")