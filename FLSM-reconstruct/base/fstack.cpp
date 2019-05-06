#include "fstack.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;
using namespace flsm;

Stack::Stack(int n) //test
{
    images.assign(n, Image("x", Mat()));
}

Stack::Stack(bool isOrigin, cv::Rect roi_)
{
    spaceing = 1;
    prefix = string("test\\test");
    suffix = string(".tif");
    endct = 0;
    origin = isOrigin;
    bound.xs = 0;
    bound.ys = 0;
    roi = roi_;
}

Stack::Stack(double spX, string Prefix, string Suffix, int start, int end, bool isOrigin, Rect roi)
{
    prefix = Prefix;
    suffix = Suffix;
    spaceing = spX;
    for(int i = start; i < end; i++){
        string fn = Prefix + to_string(i) + Suffix;
        images.push_back(Image(fn, Mat(), roi));
    }
    for(int i = start; i > end; i--){
        string fn = Prefix + to_string(i) + Suffix;
        images.push_back(Image(fn, Mat(), roi));
    }
    endct = end;    
    origin = isOrigin;
}

int Stack::loadImage()
{
    int errct = 0;
    for(size_t i = 0; i < images.size(); i++){
        errct += loadImage(i);
    }
    return errct;
}

int Stack::loadImage(size_t idx)
{
    if(idx >= images.size()) return 1;
    if(fromVideo) {
        if(loadList.count(idx) > 0) return 0;
        if(!videoInitallized) {
            video = new VideoCapture(videoFile);
            if(!video->isOpened()) return 1;
            videoInitallized = true;
            videoPos = 0;
        }
        int vidx = videoAlias[idx];
        if(vidx == -1) return 1;
        if(vidx != videoPos) {
            video->set(CAP_PROP_POS_FRAMES, videoPos);
        }
        ++videoPos;
        Mat img;
        if(!video->read(img)) return 1;
        cvtColor(img, img, CV_BGR2GRAY);
        img.convertTo(img, CV_32FC1);
        cv::exp(img / 39.3846 + 4.60517, img);
        img.convertTo(images[idx].data, CV_16UC1);
    }
    else if(images[idx].load(origin)) return 1;
    loadList.insert(idx);
    return 0;
}

int Stack::loadImage(uint start, uint end)
{
    int errct = 0;
    int s = fmin(start, 0);
    int e = fmin(end, images.size());
    for(int i = s; i < e; i++){
        errct += loadImage(i);
    }
    return errct;
}

int Stack::loadImage(vector<size_t> ilist)
{
    int errct = 0;
    for(size_t i = 0; i < ilist.size(); i++){
        errct += loadImage(i);
    }
    return errct;
}

void Stack::release()
{
    //while(loadList.size() > 0){
    //    images[*(loadList.begin())].release(origin);
    //    loadList.erase(*(loadList.begin()));
    //}
    for(auto &i : images) i.release(0);
    loadList.clear();
}

void Stack::release(size_t idx)
{
    if(idx < images.size()){
        images[idx].release(origin);
        loadList.erase(idx);
    }
}

void Stack::release(size_t start, size_t end)
{
    int s = fmin(start, 0);
    int e = fmax(end, images.size());
    for(int i = s; i < e; i++){
        images[i].release(origin);
        loadList.erase(i);
    }
}

void Stack::release(vector<size_t> rlist)
{
    for(size_t i = 0; i < rlist.size(); i++){
        if(rlist[i] < images.size()){
            images[rlist[i]].release(origin);
            loadList.erase(rlist[i]);
        }
    }
}

int Stack::save()
{
    int errct = 0;
    for(size_t i = 0; i < images.size(); i++){
        errct+=save(i);
    }
    return errct;
}

int Stack::save(uint idx)
{
    return images[idx].save();
}

Mat Stack::getImageData(size_t idx)
{
    if(idx < images.size()){
        if(loadList.count(idx) == 0) {
            if(loadImage(idx) == 0)
                return images[idx].data;
        }
        else return images[idx].data;
    }
    if(allow_nulldata) return Mat::zeros(roi.height, roi.width, CV_16UC1);
    else return Mat();
}

Point2i Stack::pictureSize()
{
    if(!origin) return Point2i(0, 0);
    if(images.empty()) return Point2i(0, 0);
    if(!images[0].isLoaded()) return Point2i(0, 0);
    return Point2i(images[0].data.cols, images[0].data.rows);
}

void Stack::append(Mat img)
{
    images.push_back(Image(prefix + to_string(endct) + suffix, img));
    endct++;
}

void Stack::append(Image img)
{
    images.push_back(img);
    endct++;
}

Image &Stack::operator[](size_t idx)
{
    return images[idx];
}

vector<Mat> Stack::genVoxelData(double scale, Point2d xRange, Point2d yRange, Point2d zRange, double pixelSize)
{
    if(images.size() == 0) return vector<Mat>();
    Mat img = getImageData(0);
    Mat buf;
    Range ry = Range(fmax((yRange.x - bound.ys) / pixelSize, 0), fmin((yRange.y - bound.ys) / pixelSize, img.cols));
    Range rz = Range(fmax(zRange.x * sqrt(2) / pixelSize, 0), fmin((zRange.y * sqrt(2) / pixelSize), img.rows));
    resize(img(rz, ry), buf, Size(0, 0), scale, scale);
    int cols = buf.cols, rows = buf.rows;
    int xs = (xRange.x - bound.xs + zRange.x) / spaceing;
    int xe = (xRange.y - bound.xs + zRange.y) / spaceing;
    uint iStart, iEnd;
    if(spaceing > 0) {
        iStart = max(xs, 0);
        iEnd = min(xe, int(images.size()));
    }
    else {
        iStart = min(xs, int(images.size()));
        iEnd = max(xe, 0);
    }
    double offset = iStart * spaceing - xRange.x + bound.xs - zRange.x;

    vector<Mat> vol;
    for(int i = 0; i < cols; i++)
        vol.push_back(Mat(rows, abs(iEnd - iStart), buf.type()));
    for(uint i = iStart; i != iEnd; i += (spaceing > 0 ? 1 : -1)) {
        img = getImageData(i);
        resize(img(rz, ry), buf, Size(0, 0), scale, scale);
        for(int j = 0; j < cols; j++){
            buf.col(j).copyTo(vol[j].col(abs(i - iStart)));
        }
        this->release(i);
        //debug output
        if(i % 100 == 0) cout << i << std::flush;
        else if (i % 10 == 0) cout << "." << std::flush;
    }
    cout << endl;

    vector<Point2f> pp({Point2f(0, 0), Point2f(1, 0), Point2f(0, 1)});
    vector<Point2f> np({Point2f(offset * scale / pixelSize, 0),
                        Point2f((offset + abs(spaceing)) * scale / pixelSize, 0),
                        Point2f(offset * scale / pixelSize - sqrt(0.5), sqrt(0.5))});
    Mat m = getAffineTransform(pp, np);

    vector<Mat> dst;
    for(size_t i = 0; i < vol.size(); i++) {
        dst.push_back(vol[i]);
        warpAffine(vol[i], dst[i], m,
                   Size((xRange.y - xRange.x) / pixelSize * scale,
                        (zRange.y - zRange.x) / pixelSize * scale),
                   INTER_CUBIC);
        //debug output
        if(i % 100 == 0) cout << i << std::flush;
        else if (i % 10 == 0) cout << "." << std::flush;
    }
    cout << endl;
    return dst;
}

void Stack::Stack::genVoxelData_(vector<Mat>& dst, double scale, Point2d xRange, Point2d yRange, Point2d zRange, double pixelSize)
{
    vector<Mat> dd = genVoxelData(scale, xRange, yRange, zRange, pixelSize);
    dst = dd;
}

Stack::~Stack()
{
    release();
}


