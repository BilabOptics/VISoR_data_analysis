#ifndef FBRAIN_H
#define FBRAIN_H
#include "fslice.h"

namespace flsm {

struct Brain
{
    Brain(std::string Name = std::string());
    void getBrainProperties(rapidjson::Value& v);

    std::string name;
    std::vector<Slice> slices;
};

struct pointer
{
    Brain *brain;
    int sl;
    int st;
    int im;

    pointer(Brain* brain_ = nullptr,
                   int slice_ = -1,
                   int stack_ = -1,
                   int image_ = -1);
    bool operator<(const pointer other) const;
    Slice *slice() const;
    Stack *stack() const;
    Image *image() const;
    int numSlice() const;
    int numStack() const;
    int numImage() const;
    void print();
    bool isValid(bool strict = true);
    void setValid(bool strict = true);
};

cv::Point3d convertPos(cv::Point2d pos2d, pointer p);
void inverseConvertPos(cv::Point3d pos3d, pointer& p, cv::Point2d& pos2d);
}

#endif // FBRAIN_H
