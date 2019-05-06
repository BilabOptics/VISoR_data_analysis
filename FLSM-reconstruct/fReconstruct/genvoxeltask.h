#ifndef GENVOXELTASK_H
#define GENVOXELTASK_H

#include "ftask.h"

namespace flsm {

class GenVoxelTask : public Task
{
public:
    GenVoxelTask();
    bool begin(processPointer p);
    bool end(processPointer p);
    bool sliceBegin(processPointer p);
    bool sliceEnd(processPointer p);
    bool stackBegin(processPointer p);
    bool stackEnd(processPointer p);
    bool loadImage(processPointer p);
    bool final(processPointer p);
    void setParameters(rapidjson::Value& v);

    double scale = 0.0625;
    box range;
    std::string savePath;

    std::vector<std::vector<int> > splitL;
    std::vector<std::vector<int> > splitR;
    std::vector<std::vector<int> > snapshotIdx;
    std::vector<std::vector<std::vector<cv::Mat> > > stackImage;
    sliceData<flsm::Stack> sliceImage;
    std::vector<bool> snapshotGenerated;

    std::vector<flsm::Stack> result;
};
}

#endif // GENVOXELTASK_H
