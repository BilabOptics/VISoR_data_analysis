from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.ensemble import AdaBoostClassifier
import csv
import numpy as np
import cv2
import matplotlib.pyplot as plt

tfile = "D:/chaoyu/cfos/validation/train.csv"
qfile = "D:/chaoyu/cfos/validation/test.csv"
savfile = "D:/chaoyu/cfos/validation/_.csv"


def refineData(srcfile):
    src = open(srcfile, "r")
    reader = csv.reader(src)
    cells = []
    for line in reader:
        cells.append(line)
    head = cells[0]
    cells.pop(0)
    print(len(cells))
    features = []
    marks = []
    cells_ = []
    for c in cells:
        if len(c) == 0:
            continue
        ff = []
        for f in c[3:len(c) - 2]:
            if not np.isnan(float(f)):
                ff.append(float(f))
            else:
                ff.append(0)
        cells_.append(c)
        features.append(ff)
        marks.append(int(c[len(c) - 1]))
    return head, cells_, features, marks


def saveClassifiedData(dstfile, head, cells, type):
    dst = open(dstfile, "w")
    writer = csv.writer(dst)
    writer.writerow(head)
    for i in range(0, len(cells)):
        line = cells[i]
        line[len(line) - 1] = str(type[i])
        writer.writerow(line)


def plotHist(data, num_bins = 30, range_=(-4, -1)):
    fig, ax = plt.subplots()
    ax_ = ax.twinx()

    for i in range(3):
        data[i] = -np.clip(data[i], 0, 249)
    hist = [np.histogram(data[i], num_bins, range_)[0] for i in range(3)]
    bin = np.histogram(data[0], num_bins, range_)[1]
    weight = []
    f = 0
    t = 0
    for i in range(len(hist[0])):
        f += hist[1][i] + hist[0][i]
        t += hist[2][i]
        weight.append(t / (f + t))
    #ax.bar(bin, hist[0])
    n, bins, patches = ax.hist([np.array(data[0]), np.array(data[1]), np.array(data[2])], num_bins, range=range_, stacked=True, rwidth=0.5, cumulative=True)
    ax_.scatter(bin[0:num_bins], weight, c='r')
    print(weight)
    print(bin)

    ax.set_xlabel('Intensity')
    ax.set_ylabel('Cell NO.')
    #ax.set_title('Histogram')

    ax_.set_ylabel('Correct rate')
    ax_.set_ylim(0, 1)

    fig.tight_layout()
    plt.show()


def train(srcfile):
    h, _, tf, tm = refineData(srcfile)
    sc = preprocessing.StandardScaler()
    sc.fit(tf)
    st = sc.transform(tf)
    model = svm.SVC(kernel="rbf", class_weight={1:1, 2:1})
    model.fit(st, tm)
    return model, sc


def classify(model, srcfile, dstfile, scaler = None):
    h, c, qf, qm = refineData(srcfile)
    if not scaler is None:
        qf = scaler.transform(qf)
    res = model.predict(qf)
    saveClassifiedData(dstfile, h, c, res)


def compareChannels(src1, src2, dstfile, classified=True, c_class=1):
    h1, c1, f1, m1 = refineData(src1)
    h2, c2, f2, m2 = refineData(src2)
    if classified == True:
        c1 = [c1[i] for i in range(len(c1)) if m1[i] == c_class]
        c2 = [c2[i] for i in range(len(c1)) if m2[i] == c_class]
    tpos = []
    qpos = []
    for c in c1:
        tpos.append([float(c[0]), float(c[1]), float(c[2])])
    for c in c2:
        qpos.append([float(c[0]), float(c[1]), float(c[2])])

    index_params = dict(algorithm=0, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    matcher = cv2.FlannBasedMatcher(index_params, search_params)
    matches = matcher.match(np.float32(qpos), np.float32(tpos))
    for m in matches:
        if m.distance > 16:
            continue



if __name__ == "__main__":
    h, _, tf, tm = refineData(tfile)
    h, cells, qf, qm = refineData(qfile)

    sc = preprocessing.StandardScaler()
    sc.fit(tf)
    st = sc.transform(tf)
    sq = sc.transform(qf)
    model = svm.SVC(kernel="rbf", class_weight={1:1, 2:1})
    model.fit(st, tm)
    res = model.predict(sq)
    print(res)

    tp = []
    tn = []
    fp = []
    fn = []
    for i in range(len(qf)):
        if qm[i] < 1.5 and res[i] < 1.5:
            tp.append(qf[i])
        elif qm[i] < 1.5:
            fn.append(qf[i])
        elif res[i] < 1.5:
            fp.append(qf[i])
        else:
            tn.append(qf[i])
    print(len(tp))
    print(len(tn))
    print(len(fp))
    print(len(fn))

    tf1 = []
    tf2 = []
    for i in range(len(qf)):
        if qm[i] < 1.5:
            tf1.append(qf[i])
        else:
            tf2.append(qf[i])

    ntf = [np.array(fp).transpose()[7], np.array(fn).transpose()[7], np.array(tp).transpose()[7]]
    plotHist(ntf)

    saveClassifiedData(savfile, h, cells, res)
