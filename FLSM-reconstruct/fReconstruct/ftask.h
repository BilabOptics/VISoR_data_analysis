#ifndef FTASK_H
#define FTASK_H
#include <set>
#include <rapidjson/document.h>
#include <queue>

#include "fbrain.h"

namespace flsm {

class Task;

struct processPointer :public pointer
{
    int step;
    int part;
    Task* task;

    processPointer(Task* task_ = nullptr,
                   Brain* brain_ = nullptr,
                   int slice_ = -1,
                   int stack_ = -1,
                   int step_ = -1,
                   int part_ = -1,
                   int image_ = -1);
    bool operator<(const processPointer other) const;
    void print();
};

class Task
{
public:
    virtual bool begin(processPointer p) = 0;
    virtual bool end(processPointer p) = 0;
    virtual bool sliceBegin(processPointer p) = 0;
    virtual bool sliceEnd(processPointer p) = 0;
    virtual bool stackBegin(processPointer p) = 0;
    virtual bool stackEnd(processPointer p) = 0;
    virtual bool loadImage(processPointer p) = 0;
    virtual bool final(processPointer p) = 0;
    virtual void setParameters(rapidjson::Value& v) = 0;
};

template <class T>
class sliceData
{
public:
    std::vector<T> data;
    sliceData();
    sliceData(Brain* br);
    Brain* brain;
    void assign(T value);
    T& operator[](processPointer p);
    T& operator[](int idx);
};

template <class T>
class stackData
{
public:
    std::vector<std::vector<T> > data;
    Brain* brain;
    stackData();
    stackData(Brain* br);
    void assign(T value);
    T& operator[](processPointer p);
    std::vector<T>& operator[](int idx);
};

class nullTask : public Task
{
public:
    nullTask();
    bool begin(processPointer p) override;
    bool end(processPointer p) override;
    bool sliceBegin(processPointer p) override;
    bool sliceEnd(processPointer p) override;
    bool stackBegin(processPointer p) override;
    bool stackEnd(processPointer p) override;
    bool loadImage(processPointer p) override;
    bool final(processPointer p) override;
    void setParameters(rapidjson::Value& v) override {}
};

class TaskManager
{
    void addPointer(processPointer p);
    void runThread();
    void ioThread();
    std::queue<processPointer> ioRequest;
public:
    TaskManager();
    void addProcess(Brain* brain);
    std::vector<Task*> tasks;
    std::multiset<processPointer> pool;
    std::multiset<processPointer> ongoing;
    std::multiset<processPointer> finished;
    void run();
};

class Process
{
    pointer position;

};

template <typename T>
sliceData<T>::sliceData()
{

}

template <typename T>
sliceData<T>::sliceData(Brain *br)
{
    brain = br;
}

template <typename T>
void sliceData<T>::assign(T value)
{
    data.assign(brain->slices.size(), value);
}

template <typename T>
T &sliceData<T>::operator[](processPointer p)
{
    return data[p.sl];
}

template <typename T>
T &sliceData<T>::operator[](int idx)
{
    return data[idx];
}

template <typename T>
stackData<T>::stackData()
{

}

template <typename T>
stackData<T>::stackData(Brain *br)
{
    brain = br;
}

template <typename T>
void stackData<T>::assign(T value)
{
    data.assign(brain->slices.size(), std::vector<T>());
    for(uint i = 0; i < brain->slices.size(); i++)
        data[i].assign(brain->slices[i].stacks.size(), value);
}

template <typename T>
T &stackData<T>::operator[](processPointer p)
{
    return data[p.sl][p.st];
}

template <typename T>
std::vector<T> &stackData<T>::operator[](int idx)
{
    return data[idx];
}
}
#endif // FTASK_H
