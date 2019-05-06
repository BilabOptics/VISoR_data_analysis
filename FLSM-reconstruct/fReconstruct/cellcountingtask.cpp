#include "cellcountingtask.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/photo.hpp>
#include <opencv2/flann.hpp>
#include <vector>
#include <iostream>
#include <fstream>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#include "mark.h"

using namespace std;
using namespace cv;

namespace flsm {

CellCountingTask::CellCountingTask()
{

}

bool CellCountingTask::begin(flsm::processPointer p)
{
    cells.brain = p.brain;
    activeObjects.brain = p.brain;
    cells.assign(vector<Structure>());
    activeObjects.assign(nullptr);
    return 0;
}

bool CellCountingTask::end(processPointer p)
{
    return 0;
}

bool CellCountingTask::sliceBegin(processPointer p)
{
    /*double pixScale = 0.125 / p.slice()->pixelSize;
    p.slice()->snapshotPath = saveDir + "/" + p.slice()->brainName + "/"
            + to_string(p.slice()->idxZ) + "-" + p.slice()->name + "-" + p.slice()->captureTime + "/";
    p.slice()->snapshot = Stack(1, p.slice()->snapshotPath, string(".tif"), 0, 0);*/
    return 0;
}

bool CellCountingTask::sliceEnd(processPointer p)
{
    double pixScale = 0.125 / p.slice()->pixelSize;
    double pye = p.slice()->bound.ys;
    int cols = ceil(pixScale * (p.slice()->bound.xe - p.slice()->bound.xs));
    int rows = ceil(pixScale * (p.slice()->bound.ye - p.slice()->bound.ys));
    Mat zProj = Mat::zeros(rows, cols, CV_16UC1);
    vector<Structure> validCells;
    for(p.st = 0; p.st < p.slice()->stacks.size(); p.st++) {
        for(Structure& cell : cells[p]) {
            if(cell.segments.size() > 1) {
                Point3d cc = cell.center();
                if(cc.y < pye) continue;
                vector<Structure> vscell;
                if(cell.segments.size() == 2) vscell.push_back(cell);
                else {
                    vector<int> pits = cell.pits();
                    if(pits.size() == 0)
                        vscell.push_back(cell);
                    else { //split structure
                        int pk = 0;
                        for(uint i = 0; i < cell.segments.size();) {
                            Structure scell(cell.segments[i++]);
                            for(; i <= pits[pk] && i < cell.segments.size(); ++i) {
                                scell.append(cell.segments[i]);
                            }
                            vscell.push_back(scell);
                            ++pk;
                        }
                    }
                }

                for(auto sc : vscell) {
                    double its = sc.intensity();
                    if(sc.segments.size() == 2) {
                        int a = 0, b = 1;
                        if(sc.segments[0].intensity > sc.segments[1].intensity) {
                            a = 1;
                            b = 0;
                        }
                        Structure c1(sc.segments[a]);
                        c1.append(cell.segments[b]);
                        c1.append(cell.segments[a]);
                        c1.segments[2].features[a] = 0;
                        c1.segments[2].features[b] = 0;
                    }

                    validCells.push_back(sc);
                    cc = sc.center();

                    cc.x = (cc.x - p.slice()->bound.xs) * pixScale;
                    cc.y = (cc.y - p.slice()->bound.ys) * pixScale;
                    cc.z = (cc.z - p.slice()->bound.zs) * pixScale;
                    zProj.at<uint16_t>(round(cc.y), round(cc.x)) += its;
                }
            }
        }
        pye = p.stack()->bound.ye;
    }
    string fn = p.slice()->snapshotPath + "cells.csv";
    fstream fs(fn, fstream::trunc | fstream::out);
    if(!fs.is_open()) cout << "Fail to open " << fn << endl;
    else {
        fs << '"' << p.slice()->name << "\"," <<
              p.slice()->idxZ << ',' <<
              p.slice()->bound.xs << ',' <<
              p.slice()->bound.ys << ",,,,,,,,,,,,,,,,,,,,,,,,,,,,,," << endl;
        for(auto cell : validCells) {
            mark mk(cell);
            int pk = cell.peak();
            if(pk < 1 || pk > cell.segments.size() - 2) continue;
            fs << mk.center.x << ',' <<
                  mk.center.y << ',' <<
                  mk.center.z << ',' <<
                  cell.shift(pk, pk - 1) << ',' <<
                  cell.shift(pk, pk + 1) << ',' <<
                  cell.shift(pk - 1, pk + 1) << ',';
            for(int i = -1; i < 2; i++) {
                for(double feature : cell.segments[pk + i].features) {
                    fs << feature << ',';
                }
            }
            fs << "0,0" << endl;
        }
        fs.close();
    }
    fn = p.slice()->snapshotPath + "cellmarks.tif";
#ifdef _WIN32
    for(char& ch : fn) {
        if(ch == '/') ch = '\\';
    }
#endif
    Image i(fn, zProj);
    i.save();
    for(p.st = 0; p.st < p.slice()->stacks.size(); p.st++)
        cells[p].clear();
    return 0;
}

bool CellCountingTask::stackBegin(processPointer p)
{
    cells[p].reserve(1000000);//TODO: Dynamic memory allocation.
    return 0;
}

bool CellCountingTask::stackEnd(processPointer p)
{
    return 0;
}

bool CellCountingTask::loadImage(processPointer p)
{
    //p.print();
    //if(p.im < 200 || p.im > 2500) return 0;
    Mat img = p.stack()->getImageData(p.im);
    vector<float> h(1, 3.0);
    //fastNlMeansDenoising(img, img, h, 7, 7, NORM_L1);
    GaussianBlur(img, img, Size(0, 0), 3);
    Mat im4, im4i;
    pyrDown(img, im4i);
    pyrDown(im4i, im4i);
    im4i.convertTo(im4, CV_32FC1);
    im4 = im4 - p.slice()->brightness * 100 - p.slice()->background;
    Mat dog;
    GaussianBlur(im4, dog, Size(0, 0), cellRadius / 4);
    dog = im4 - dog;
    vector<Point> maxima = findMaxima(dog, cellRadius / 4, 5);
    vector<Segment> vseg;
    vector<Point> new_maxima;
    for(Point mp : maxima) {
        mp.x *= 4;
        mp.y *= 4;
        if(mp.x < 4 || mp.y < 4 || mp.x + 5 > img.cols || mp.y + 5 > img.rows)
            continue;
        Point ps;
        minMaxLoc(img(Rect(mp.x - 4, mp.y - 4, 9, 9)), 0, 0, 0, &ps);
        mp.x += ps.x - 4;
        mp.y += ps.y - 4;
        if(mp.x < cellRadius * 3 || mp.y < cellRadius * 3 ||
                mp.x + cellRadius * 3 + 1 > img.cols || mp.y + cellRadius * 3 + 1 > img.rows)
            continue;
        new_maxima.push_back(mp);
    }
    vector<vector<Point2i> > neibours = distSearch(new_maxima, cellRadius * 3, cellRadius);
    for(uint i = 0; i < new_maxima.size(); ++i){
        Point mp = new_maxima[i];
        Segment seg(p);
        seg.radius = cellRadius;
        seg.center = mp;
        seg.extract(img(Rect(mp.x - 3 * cellRadius, mp.y - 3 * cellRadius, 6 * cellRadius + 1, 6 * cellRadius + 1)), neibours[i]);
        if(seg.confidence() > 0 && seg.valid)
            vseg.push_back(seg);
    }/*
    Mat db;
    cvtColor(im4i, db, CV_GRAY2BGR);
    db.convertTo(db, CV_8UC3, 0.2);
    for(auto seg : vseg) {
        auto p = Point(seg.center.x/4, seg.center.y/4);
        line(db,Point(p.x - 1, p.y),Point(p.x + 1, p.y), Scalar(0, 255, 0));
        line(db,Point(p.x, p.y - 1),Point(p.x, p.y + 1), Scalar(0, 255, 0));
    }
    imshow("s" ,db);
    waitKey(5);*/
    const double shift = p.stack()->spaceing / p.slice()->pixelSize / sqrt(2);
    if(vseg.size() == 0) {
        delete activeObjects[p];
        activeObjects[p] = nullptr;
        //p.stack()->release();
        //return 0;
    }
    else if(activeObjects[p] != nullptr) {
        vector<Structure*>* act = new vector<Structure*>();
        Mat tdesc(activeObjects[p]->size(), 2, CV_32FC1), qdesc(vseg.size(), 2, CV_32FC1);
        for(uint i = 0; i < activeObjects[p]->size(); ++i) {
            tdesc.at<float>(i ,0) = (*activeObjects[p])[i]->last().center.x;
            tdesc.at<float>(i, 1) = shift + (*activeObjects[p])[i]->last().center.y ;
        }
        for(uint i = 0; i < vseg.size(); ++i) {
            qdesc.at<float>(i, 0) = vseg[i].center.x;
            qdesc.at<float>(i, 1) = vseg[i].center.y;
        }
        vector<vector<DMatch> > matches;
        FlannBasedMatcher matcher;
        matcher.radiusMatch(qdesc, tdesc, matches, cellRadius);
        for(uint i = 0; i < vseg.size(); ++i) {
            bool unm = true;
            for(auto dm : matches) {
                if(dm.size() == 0) continue;
                if(dm[0].queryIdx == i) {
                    (*activeObjects[p])[dm[0].trainIdx]->append(vseg[i]);
                    act->push_back((*activeObjects[p])[dm[0].trainIdx]);
                    unm = false;
                    break;
                }
            }
            if(unm) {
                cells[p].push_back(Structure(vseg[i]));
                act->push_back(&(cells[p][cells[p].size() - 1]));
            }
        }/*
        int qct = 0;
        for(auto dm : matches) {
            while(qct < dm[0].queryIdx) {
                cells[p].push_back(Structure(vseg[qct]));
                act->push_back(&(cells[p][cells[p].size() - 1]));
                ++qct;
            }
            if(dm.size() > 0) {
                if (dm[0].distance < 2 * cellRadius){
                    (*activeObjects[p])[dm[0].trainIdx]->append(vseg[dm[0].queryIdx]);
                    act->push_back((*activeObjects[p])[dm[0].trainIdx]);
                    continue;
                }
            }
            cells[p].push_back(Structure(vseg[dm[0].queryIdx]));
            act->push_back(&(cells[p][cells[p].size() - 1]));
        }*/
        delete activeObjects[p];
        activeObjects[p] = act;
    }
    else {
        activeObjects[p] = new vector<Structure*>();
        for(Segment seg : vseg) {
            cells[p].push_back(Structure(seg));
            activeObjects[p]->push_back(&(cells[p][cells[p].size() - 1]));
        }
    }
    /*Mat db;
    cvtColor(im4i, db, CV_GRAY2RGB);
    if(activeObjects[p] != nullptr) {
        for(Structure* s : *activeObjects[p]) {
            circle(db, Point(s->last().center.x / 4, s->last().center.y / 4), 2, Scalar(1000, 0, 0));
        }
    }
    //imshow("s", db * 100);
    //waitKey(5);*/
    p.stack()->release();
    return 0;
}

bool CellCountingTask::final(processPointer p)
{
    return 0;
}

void CellCountingTask::setParameters(rapidjson::Value &v)
{
    if(!v.IsObject()) return;
    if(v.HasMember("cellRadius"))
        if(v["cellRadius"].IsDouble())
            cellRadius = v["cellRadius"].GetDouble();
    if(v.HasMember("saveDir"))
        if(v["saveDir"].IsString()) {
            saveDir = string(v["saveDir"].GetString());
#ifdef _WIN32
            for(char& ch : saveDir) {
                if(ch == '/') ch = '\\';
            }
#endif
        }
}

std::vector<Point2i> findMaxima(Mat src, double radius, double thresh)
{
    //double diam = radius * 2 + 1;
    Mat wei, thr;
    dilate(src, wei, Mat::ones(5, 5, CV_8UC1));
    absdiff(wei, src, wei);
    threshold(wei, thr, 0.00001, 1, THRESH_BINARY_INV);
    /*
    gaussianDiff(src, thr, diam);
    threshold(-thr, thr, 0, 1, THRESH_TOZERO);
    gaussianDiff(src + thr, thr, diam);
    multiply(dil, thr, dil);
    */
    blur(wei, wei, Size(3, 3));
    multiply(wei, thr, thr);
    threshold(thr, thr, thresh, 255, THRESH_BINARY);
    thr.convertTo(thr, CV_8UC1);
    vector<vector<Point> > ct;
    findContours(thr, ct, RETR_LIST, CHAIN_APPROX_SIMPLE);
    vector<Point> plist;
    for(vector<Point> t : ct) {
        int x = 0, y = 0;
        for(Point p : t) {
            x += p.x;
            y += p.y;
        }
        plist.push_back(Point(x / t.size(), y / t.size()));
    }
    return plist;
}

void gaussianDiff(Mat src, Mat& dst, double diam)
{
    GaussianBlur(src, dst, Size(0, 0), diam);
    dst = src - dst;
}

}
