#ifndef STRUCTURE_H
#define STRUCTURE_H

#include <opencv2/core.hpp>
#include <vector>

#include "fstack.h"
#include "segment.h"

namespace flsm {

struct Structure
{
public:
    Structure(Segment first);
    double match(Segment next);
    void append(Segment next);
    double correctness();
    cv::Point3d center();
    double intensity();
    double totalIntensity();
    double shift(int i, int j);
    int peak();
    std::vector<int> peaks();
    std::vector<int> pits();
    Segment& last();

    std::vector<Segment> segments;
    pointer position;
};

struct StructureMark
{
public:
    StructureMark();
    StructureMark(Structure src);
    double correctness;
    cv::Point3d center;
    double intensity;
};

//cv::Point3d convertPos(cv::Point2d p2d, pointer pos);

}
#endif // STRUCTURE_H
