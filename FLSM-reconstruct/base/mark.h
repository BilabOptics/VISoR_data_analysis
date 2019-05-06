#ifndef MARK_H
#define MARK_H
#include <opencv2/core.hpp>
#include <vector>
#include "structure.h"

namespace flsm {

const int mktype_default = 0;
const int mktype_true = 1;
const int mktype_false = 2;

class mark
{
public:
    mark();
    mark(Structure str, int group_ = 0);
    std::vector<double> features;
    cv::Point3d center;
    int type;
    int group;
};

}
#endif // MARK_H
