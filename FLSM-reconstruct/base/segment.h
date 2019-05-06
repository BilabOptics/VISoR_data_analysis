#ifndef SEGMENT_H
#define SEGMENT_H
#include <opencv2/core.hpp>
#include <vector>

#include "fbrain.h"

namespace flsm {

struct Segment
{
public:
    Segment(pointer p);
    double confidence();
    double correctness();
    void extract(cv::Mat src, std::vector<cv::Point2i> neighbors = std::vector<cv::Point2i>());

    std::vector<double> features;
    double radius = 0;
    double area = 0;
    double circularity = 0;
    double intensity = 0;
    double eccentricity = 0;
    cv::RotatedRect ellipse;
    cv::Point2i center;

    std::vector<cv::Point2i> contour;
    pointer position;
    bool valid = false;
};

}
#endif // SEGMENT_H
