#ifndef CELLCOUNTINGTASK_H
#define CELLCOUNTINGTASK_H
#include <list>
#include <string>

#include "ftask.h"
#include "structure.h"

namespace flsm {

class CellCountingTask : public Task
{
public:
    CellCountingTask();
    bool begin(processPointer p);
    bool end(processPointer p);
    bool sliceBegin(processPointer p);
    bool sliceEnd(processPointer p);
    bool stackBegin(processPointer p);
    bool stackEnd(processPointer p);
    bool loadImage(processPointer p);
    bool final(processPointer p);
    void setParameters(rapidjson::Value& v);

    stackData<std::vector<Structure> > cells;
    stackData<std::vector<Structure*>* > activeObjects;
    double cellRadius = 14;
    std::string saveDir;

};

std::vector<cv::Point2i> findMaxima(cv::Mat src, double radius, double thresh);
void gaussianDiff(cv::Mat src, cv::Mat &dst, double diam);
}
#endif // CELLCOUNTINGTASK_H
