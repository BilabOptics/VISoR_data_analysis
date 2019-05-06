#ifndef COMPRESSTASK_H
#define COMPRESSTASK_H

#include <ftask.h>

namespace flsm {

class CompressTask : public Task
{
public:
    CompressTask();
    bool begin(processPointer p);
    bool end(processPointer p);
    bool sliceBegin(processPointer p);
    bool sliceEnd(processPointer p);
    bool stackBegin(processPointer p);
    bool stackEnd(processPointer p);
    bool loadImage(processPointer p);
    bool final(processPointer p);
    void setParameters(rapidjson::Value& v);
};

}
#endif // COMPRESSTASK_H
