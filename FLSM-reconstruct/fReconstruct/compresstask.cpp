#include "compresstask.h"

using namespace flsm;

CompressTask::CompressTask()
{

}

bool CompressTask::begin(processPointer p)
{

}

bool CompressTask::end(processPointer p)
{

}

bool CompressTask::sliceBegin(processPointer p)
{

}

bool CompressTask::sliceEnd(processPointer p)
{

}

bool CompressTask::stackBegin(processPointer p)
{

}

bool CompressTask::stackEnd(processPointer p)
{

}

bool CompressTask::loadImage(processPointer p)
{

}

bool CompressTask::final(processPointer p)
{

}

void flsm::CompressTask::setParameters(rapidjson::Value &v)
{
    if(!v.IsObject()) return;
}
