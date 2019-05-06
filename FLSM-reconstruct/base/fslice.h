#ifndef FSLICE_H
#define FSLICE_H
#include <string>
#include <vector>

#include "fstack.h"

namespace flsm {

struct Slice
{
    int idxZ;
    double pixelSize;
    box bound;
    cv::Mat transform;

    std::string idxFile;
    std::string name;
    std::string captureTime;
    std::string brainName;
    std::vector<Stack> stacks;
    bool initiallized = false;
    Stack snapshot;
    std::string snapshotPath;
    bool snapshotInitiallized = false;
    double background = 100;
    double brightness = 1;
    double noise = 5;
    box roi;

    Slice();//test
    Slice(std::string idxFileName, int idx_z = 0, double pixSize = 1, bool useVideo = true, box roi = box());
    void genVoxelData(double scale, Stack &dst, cv::Point2d xRange, cv::Point2d yRange, cv::Point2d zRange);
};

}

#endif // FSLICE_H
