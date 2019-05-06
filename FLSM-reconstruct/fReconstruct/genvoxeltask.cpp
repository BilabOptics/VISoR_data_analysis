#include "genvoxeltask.h"
#include <iostream>
#include <cmath>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;
using namespace flsm;

void flatten(Slice* src, Stack& dst, Mat& abo, Mat& und, double output_scale)
{
    const double base_embedding_bri = 4.9,
            bri_difference = 0.0,
            base_sample_bri = 5.1;
    double embedding_bri = base_embedding_bri + log(src->brightness);
    double sample_bri = base_sample_bri + log(src->brightness);
    int h = 0;
    for(Image im : src->snapshot.images){
        Mat img;
        im.data.convertTo(img, CV_32FC1);
        cv::log(img, img);
        if(h == 0) h = img.rows;
        Mat un = Mat(1, img.cols, CV_8UC1, Scalar(img.rows));
        Mat ab = Mat::zeros(1, img.cols, CV_8UC1);
        Mat sob;
        Sobel(img, sob, CV_32F, 0, 1, -1);
        for(int i = 0; i < img.cols; ++i) {
            for(int j = 0; j < img.rows - 5; ++j) {
                if(img.at<float_t>(j, i) > embedding_bri) {
                    int k = 1;
                    for(; k < 5; ++k) {
                        if(img.at<float_t>(j + k, i) < embedding_bri) break;
                    }
                    if(k != 5) continue;
                    float s = sob.at<float>(j, i);
                    int l = 0;
                    for(k = 1; k < 10; ++k) {
                        if(j + k >= img.rows) break;
                        if(img.at<float_t>(j + k, i) > sample_bri) {
                            l = k;
                            break;
                        }
                        if(sob.at<float>(j + k, i) > s) {
                            l = k;
                            s = sob.at<float>(j + k, i);
                        }
                        else if(s > 5) break;
                    }
                    ab.at<uchar>(i) = j + l;
                    break;
                }
            }
            for(int j = img.rows - 1; j > 4; --j) {
                if(img.at<float_t>(j, i) > embedding_bri - bri_difference) {
                    int k = 1;
                    for(; k < 5; ++k) {
                        if(img.at<float_t>(j - k, i) < embedding_bri - bri_difference) break;
                    }
                    if(k != 5) continue;
                    float s = sob.at<float>(j, i);
                    int l = 0;
                    for(k = 1; k < 10; ++k) {
                        if(j - k < 0) break;
                        if(img.at<float_t>(j - k, i) > sample_bri - bri_difference) {
                            l = k;
                            break;
                        }
                        if(sob.at<float>(j - k, i) < s) {
                            l = k;
                            s = sob.at<float>(j - k, i);
                        }
                        else if(s < -5) break;
                    }
                    un.at<uchar>(i) = j - l;
                    break;
                }
            }
        }
        abo.push_back(ab);
        und.push_back(un);
    }
    medianBlur(abo, abo, 5);
    medianBlur(und, und, 5);
    const double scale = output_scale;
    vector<Mat> vres;
    for(int i = 0; i < src->snapshot.images.size(); ++i) {
        Mat img = src->snapshot.images[i].data;
        Mat res = Mat::zeros(83, img.cols, img.type());
        uchar* pa = abo.ptr(i);
        uchar* pb = und.ptr(i);
        for(int j = 0; j < img.cols; j++) {
            int a = pa[j];
            int b = pb[j];
            if(b - a > 3)
                resize(img.col(j).rowRange(a, b),
                       res.col(j), res.col(j).size());
        }
        resize(res, res, Size(0, 0), scale, scale);
        vres.push_back(res);
    }
    for(int i = 0; i < vres[0].rows; ++i) {
        Mat b1(vres.size(), vres[0].cols, vres[0].type());
        for(int j = 0; j < vres.size(); ++j) {
            vres[j].row(i).copyTo(b1.row(j));
        }
        resize(b1, b1, Size(0, 0), 1, scale);
        b1.convertTo(b1, CV_32FC1);
        cv::log(b1, b1);
        b1.convertTo(b1, CV_8UC1, 39.47459, -181.788);
        dst.append(b1);
    }
}

GenVoxelTask::GenVoxelTask()
{

}

bool GenVoxelTask::begin(processPointer p)
{
    splitL.assign(p.numSlice(), vector<int>());
    splitR.assign(p.numSlice(), vector<int>());
    snapshotIdx.assign(p.numSlice(), vector<int>());
    stackImage.assign(p.numSlice(),vector< vector <Mat> >());
    sliceImage.brain = p.brain;
    sliceImage.assign(Stack());
    snapshotGenerated.assign(p.numSlice(), false);
    return 0;
}

bool GenVoxelTask::sliceBegin(processPointer p)
{
    int ct = 0;
    double ss = p.slice()->bound.ys;
    double pixelScale =  scale / p.slice()->pixelSize;
    for(uint i = 0; i < p.slice()->stacks.size(); i++){
        splitL[p.sl].push_back(max(int((ss - p.slice()->stacks[i].bound.ys) * pixelScale) - 1, 0));
        splitR[p.sl].push_back(int(floor((p.slice()->stacks[i].bound.ye - ss) * pixelScale)) + splitL[p.sl][i]);
        snapshotIdx[p.sl].push_back(ct);
        ct += splitR[p.sl][i] - splitL[p.sl][i] - 1;
        ss += (splitR[p.sl][i] - splitL[p.sl][i]) / pixelScale;
        stackImage[p.sl].assign(p.numStack(), vector<Mat>());
    }
    createFolder(p.slice()->snapshotPath + "8/");
    createFolder(p.slice()->snapshotPath + "25/");
    p.slice()->snapshot = Stack(1, p.slice()->snapshotPath + "8/", string(".tif"), 0, ct);
    sliceImage[p] = Stack(1, p.slice()->snapshotPath + "25/", string(".tif"), 0, 0);
    /*if(p.slice()->snapshot.getImageData(0).cols > 0) {
        if(p.slice()->snapshot.loadImage() > 0)
            p.slice()->snapshot.release();
        else p.slice()->snapshotInitiallized = true;
    }*/
    return 0;
}

bool GenVoxelTask::stackBegin(processPointer p)
{
    if(p.slice()->snapshotInitiallized) return 0;
    Mat img = p.stack()->getImageData(0);
    Mat buf;
    resize(img, buf, Size(0, 0), scale, scale);
    int cols = splitR[p.sl][p.st] - splitL[p.sl][p.st];
    stackImage[p.sl][p.st].reserve(cols);
    for(int i = 0; i < cols; i++) {
        stackImage[p.sl][p.st].push_back(Mat(buf.rows, p.numImage(), buf.type()));
    }
    return 0;
}

bool GenVoxelTask::loadImage(processPointer p)
{
    if(p.slice()->snapshotInitiallized) return 0;
    Mat img = p.stack()->getImageData(p.im);
    if(img.cols == 0) return -1;
    Mat buf;
    resize(img, buf, Size(0, 0), scale, scale);
    buf = buf.colRange(splitL[p.sl][p.st], splitR[p.sl][p.st]);
    const int cols = splitR[p.sl][p.st] - splitL[p.sl][p.st];
    if(p.stack()->spaceing > 0)
        for(int i = 0; i < cols; i++)
            buf.col(i).copyTo(stackImage[p.sl][p.st][i].col(p.im));
    else
        for(int i = 0; i < cols; i++)
            buf.col(i).copyTo(stackImage[p.sl][p.st][i].col(p.numImage() - p.im - 1));
    return 0;
}

bool GenVoxelTask::final(processPointer p)
{
    return 0;
}

void GenVoxelTask::setParameters(rapidjson::Value &v)
{
    if(!v.IsObject()) return;
    if(v.HasMember("scale")){
        if(v["scale"].IsDouble())
            scale = v["scale"].GetDouble();
    }
}

bool GenVoxelTask::stackEnd(processPointer p)
{
    if(p.slice()->snapshotInitiallized) return 0;
    double pixelScale =  scale / p.slice()->pixelSize;
    double offset = (min(p.stack()->bound.xs, p.stack()->bound.xe) - p.slice()->bound.xs) * pixelScale;
    vector<Point2f> pp({Point2f(0, 0), Point2f(1, 0), Point2f(0, 1)});
    vector<Point2f> np({Point2f(offset, 0),
                        Point2f(abs(p.stack()->spaceing) * pixelScale + offset, 0),
                        Point2f(offset - sqrt(0.5), sqrt(0.5))});
    Mat m = getAffineTransform(pp, np);
    for(uint i = 1; i < stackImage[p.sl][p.st].size(); i++){
        Mat dst;
        warpAffine(stackImage[p.sl][p.st][i], dst, m, Size(ceil(p.slice()->bound.xl() * pixelScale),
                                                          ceil(p.slice()->bound.zl() * pixelScale)), INTER_CUBIC);
        p.slice()->snapshot.images[snapshotIdx[p.sl][p.st] + i - 1].data = dst;
    }
    stackImage[p.sl][p.st].clear();
    return 0;
}

Rect boundaryRect(Mat src)
{
    Mat b1, bu;
    src -= 175;
    Sobel(src, b1, CV_16SC1, 1, 0, -1);
    convertScaleAbs(b1, bu, 1.0/64);
    //convertScaleAbs(src, bu, 1.0/64);
    threshold(bu, bu, 3, 255, THRESH_BINARY);
    //imshow("s", bu);
    //waitKey(0);
    vector<vector<Point> > contours;
    findContours(bu, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    if(contours.size() == 0) return Rect(0, 0, 0, 0);
    vector<Point> cc;
    for(auto c : contours) {
        for(auto p : c)
            cc.push_back(p);
    }
    return boundingRect(cc);
}

bool GenVoxelTask::sliceEnd(processPointer p)
{/*
    if(p.slice()->snapshotInitiallized) return 0;
    vector<double> z;
    double zm = 1.1;
    vector<double> o;
    vector<double> h;
    z.push_back(1);
    o.push_back(0);
    double pixelScale =  scale / p.slice()->pixelSize;
    vector<Point2f> pp({Point2f(0, 0), Point2f(1, 0), Point2f(0, 1)});
    vector<Point2f> np({Point2f(0, 0),
                        Point2f(fabs(p.slice()->stacks[0].spaceing) * pixelScale, 0),
                        Point2f(- sqrt(0.5), sqrt(0.5))});
    Mat M = getAffineTransform(pp, np);
    for(int i = 1; i < p.numStack(); i++) {
        static double zz = 1;
        static double oo = 0;
        static double hh = 0;
        Mat b11 = *(--stackImage[p.sl][i - 1].end());
        Mat b21 = stackImage[p.sl][i][0];
        Mat b1, b2;
        warpAffine(b11, b1, M, Size(p.slice()->bound.xl() * pixelScale,
                                    p.slice()->bound.zl() * pixelScale));
        warpAffine(b21, b2, M, Size(p.slice()->bound.xl() * pixelScale,
                                    p.slice()->bound.zl() * pixelScale));
        //medianBlur(b1, b1, 5);
        //medianBlur(b2, b2, 5);
        Rect r1 = boundaryRect(b1);
        Rect r2 = boundaryRect(b2);
        //Moments mo1 = moments(b1);
        //Moments mo2 = moments(b2);
        if(r1.width == 0 || r2.width == 0) {
            z.push_back(1);
            o.push_back(0);
            h.push_back(0);
        }
        else {
            zz = double(r2.width) / double(r1.width) * zz;
            z.push_back(zz);
            oo += r2.x - r1.x;
            o.push_back(oo);
            hh += r2.y - r1.y;
            h.push_back(hh);
            if(zz < zm) zm = zz;
        }
        //zz = zz * sqrt(mo2.nu20 / mo1.nu20);
        //z.push_back(zz);
        //oo += mo2.m10 / mo2.m00 - mo1.m10 / mo1.m00;
        //hh += mo2.m01 / mo2.m00 - mo1.m01 / mo1.m00;
        //o.push_back(oo);
        //h.push_back(hh);
        if(zz < zm) zm = zz;
    }
    for(int i = 0; i < p.numStack(); i++) {
        p.st = i;
        //cout << o[i] << ";";
        //cout << h[i] << ";";
        //cout << z[i] << endl;
        vector<Point2f> pp({Point2f(0, 0), Point2f(1, 0), Point2f(0, 1)});
        vector<Point2f> np({Point2f(-o[i], 0),
                            Point2f(abs(p.stack()->spaceing) * pixelScale / z[i] * zm - o[i], 0),
                            Point2f(-sqrt(0.5) - o[i], sqrt(0.5))});
        Mat m = getAffineTransform(pp, np);
        for(int j = 1; j < stackImage[p.sl][i].size(); j++){
            Mat dst;
            warpAffine(stackImage[p.sl][i][j], dst, m, Size(p.slice()->bound.xl() * pixelScale,
                                                              p.slice()->bound.zl() * pixelScale));
            p.slice()->snapshot.images[snapshotIdx[p.sl][p.st] + j - 1].data = dst;
        }
    }*/
    Stack snap = p.slice()->snapshot;
    Mat zProjection = Mat::zeros(snap.images.size(), snap.getImageData(0).cols, snap.getImageData(0).type());
    for(uint i = 0; i < p.slice()->snapshot.images.size(); i++) {
        Mat sn = p.slice()->snapshot.getImageData(i);
        Mat psn = zProjection.row(i);
        for(int j = 0; j < sn.rows; j++) {
            Mat m = cv::max(psn, sn.row(j));
            m.copyTo(psn);
        }
    }
    string fn = p.slice()->snapshotPath;
    fn.pop_back();
    fn += ".tif";
    Image i(fn, zProjection);
    i.save();
    p.slice()->snapshot.save();

    Mat abo, und;
    flatten(p.slice(), sliceImage[p], abo, und, p.slice()->pixelSize / scale / 25);
    sliceImage[p].save();
    sliceImage[p].release();
    fn = p.slice()->snapshotPath + "a.tif";
    Image a(fn, abo);
    a.save();
    fn = p.slice()->snapshotPath + "u.tif";
    Image u(fn, und);
    u.save();
    p.slice()->snapshot.release();
    return 0;
}

bool GenVoxelTask::end(processPointer p)
{
    return 0;
}
