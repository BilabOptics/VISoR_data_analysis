#include "segment.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <cmath>
#include <iostream>

using namespace flsm;
using namespace std;
using namespace cv;

Segment::Segment(pointer p)
{
    position = p;
    intensity = -100;
}

double Segment::confidence()
{
    return ((intensity - 1.01) * 1) + ((circularity - 0.7) * 1);
}

double Segment::correctness()
{
    return ((eccentricity - 0.6) * 1) + ((circularity - 0.6) * 1);
}

void Segment::extract(Mat src, vector<Point2i> neighbors)
{
    Point2i center((src.cols - 1) / 2, (src.rows - 1) / 2);
    double outside_thresh = position.slice()->background - position.slice()->brightness * 100;
    if(src.at<uint16_t>(0, (src.rows - 1) / 2) - outside_thresh < 0)
        if(src.at<uint16_t>(0, 0) - outside_thresh < 0 || src.at<uint16_t>(0, src.rows - 1) - outside_thresh < 0)
            return;
    if(src.at<uint16_t>(src.cols - 1, (src.rows - 1) / 2) - outside_thresh < 0)
        if(src.at<uint16_t>(src.cols - 1, 0) - outside_thresh < 0 || src.at<uint16_t>(src.cols - 1, src.rows - 1) - outside_thresh < 0)
            return;
    src.convertTo(src, CV_32FC1);
    radius = (src.cols - 1) / 6;
    if(radius < 1) return;
    double maxval = src.at<float>(center.y, center.x);
    Mat medm = Mat::zeros(1, 2 * (src.cols + src.rows), CV_32FC1);
    src.row(0).copyTo(medm.colRange(0, src.cols));
    src.row(src.rows - 1).copyTo(medm.colRange(src.cols, 2 * src.cols));
    transpose(src.col(0), medm.colRange(2 * src.cols, 2 * src.cols + src.rows));
    transpose(src.col(src.cols - 1), medm.colRange(2 * src.cols + src.rows, medm.cols));
    cv::sort(medm, medm, SORT_ASCENDING + SORT_EVERY_ROW);
    double medval = medm.at<float>(0, (medm.cols - 1) / 2);
    if(medval - position.slice()->background - position.slice()->brightness * 100 < 0) return;
    if(maxval < medval) return;
    Mat thr;
    threshold(src, thr, (maxval + medval) / 2, 255, THRESH_BINARY);
    thr.convertTo(thr, CV_8UC1);

    for(Point2i p : neighbors) {
        if(Point2i(p.x + center.x, p.y + center.y).inside(Rect(0, 0, src.cols, src.rows)) &&
                src.at<float>(p.y + center.y, p.x + center.x) < (maxval + medval) / 2) continue;
        int t = std::max(abs(p.x), abs(p.y));
        double its = 65535;
        Point2i vp = center;
        for(int i = 1; i < t; i++) {
            Point2i pp(floor(p.x / t) * i + center.x, floor(p.y / t) * i + center.y);
            if(!pp.inside(Rect(0, 0, src.cols, src.rows))) break;
            int it = src.at<float>(pp.y, pp.x);
            if(it < its) {
                its = it;
                vp = pp;
            }
        }
        double ac = p.y / dist(p, Point2i(0, 0));
        double as = -p.x / dist(p, Point2i(0, 0));
        if(vp.inside(Rect(0, 0, src.cols, src.rows)) && vp != Point2i(p.x + center.x, p.y + center.y)) {
            line(thr,
                 Point(vp.x + src.cols * ac, vp.y + src.rows * as),
                 Point(vp.x - src.cols * ac, vp.y - src.rows * as),
                 Scalar(0));
        }
    }

    vector<vector<Point2i> > ct;
    findContours(thr, ct, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    for(auto t : ct) {
        if(pointPolygonTest(t, Point2f((src.cols - 1) / 2, (src.rows - 1) / 2), false) >= 0) {
            if(t.size() < 5) return;
            for(auto pt : t)
                if(pt.x == 0 || pt.y == 0 || pt.x == src.cols - 1 || pt.y == src.rows - 1) return;
            contour = t;
            break;
        }
    }

    if(contour.size() == 0) return;
    double ctlenth = arcLength(contour, true);
    intensity = (maxval - position.slice()->background) / (medval - position.slice()->background);
    ellipse = fitEllipse(contour);
    area = contourArea(contour);
    circularity = (4 * 3.141592653589793 * area) / (ctlenth * ctlenth);
    eccentricity = fmin(ellipse.boundingRect().width, ellipse.boundingRect().height) /
            fmax(ellipse.boundingRect().width, ellipse.boundingRect().height);
    features.push_back(intensity);
    features.push_back(area);
    features.push_back(circularity);
    features.push_back(eccentricity);
    /*
    Mat msk = Mat::zeros(src.size(), CV_32FC1);
    fillConvexPoly(msk, contour, Scalar(1.0));
    Mat par = (src - medval) * msk;
    Moments mo = moments(par);
    vector<double> hu;
    HuMoments(mo, hu);
    for(auto h : hu) {
        features.push_back(log(h));
    }*/
    valid = true;
}

