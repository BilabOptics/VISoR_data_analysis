#include "ftask.h"
#include <iostream>
#include <map>
#include <omp.h>

using namespace std;
using namespace cv;
using namespace flsm;

nullTask::nullTask()
{

}

bool nullTask::begin(processPointer p)
{
    return testRun(100, 150);
}

bool nullTask::end(processPointer p)
{
    return testRun(100, 150);
}

bool nullTask::sliceBegin(processPointer p)
{
    return testRun(100, 150);
}

bool nullTask::sliceEnd(processPointer p)
{
    return testRun(100, 150);
}

bool nullTask::stackBegin(processPointer p)
{
    return testRun(100, 150);
}

bool nullTask::stackEnd(processPointer p)
{
    return testRun(100, 150);
}

bool nullTask::loadImage(processPointer p)
{
    if(p.im % 10 == 0)
        testRun(1, 2);
    return 0;
}

bool nullTask::final(processPointer p)
{
    return testRun(100, 150);
}

TaskManager::TaskManager()
{

}

void TaskManager::addProcess(Brain * brain)
{
    for(size_t i = 0; i < tasks.size(); i++) {
        pool.insert(processPointer(tasks[i], brain));
    }
}

void TaskManager::addPointer(processPointer p)
{
#pragma omp critical(pool)
    pool.insert(p);
}

void TaskManager::run()
{
#pragma omp parallel
    {
        runThread();
    }
}

void TaskManager::runThread()
{
    processPointer pp;
    multiset<processPointer>::iterator ip;

    while(pool.size() != 0 || ongoing.size() != 0){

        bool wait_ = false;
#pragma omp critical(pool)
        {
            ip = pool.begin();
            while((*ip).im == 0 && ip != pool.end()){
                if(pool.count(*ip) == tasks.size())
                    break;
                ip++;
            }
            if(ip != pool.end()){
                pp = *ip;
                pool.erase(pp);
            }
            else wait_ = true;
        }
        if(wait_){
            this_thread::sleep_for(chrono::milliseconds(10));
            continue;
        }

#pragma omp critical(ongoing)
        {
            ongoing.insert(pp);
        }
#pragma omp critical(io)
        {
            pp.print();
        }
        if(pp.sl == -1) {
            pp.task->begin(pp);
            for(int i = 0; i < pp.numSlice(); i++)
                addPointer(processPointer(pp.task, pp.brain, i));
        }
        else if(pp.sl == pp.numSlice()) {
            pp.task->end(pp);
        }
        else if(pp.st == -1) {
            if(!pp.task->sliceBegin(pp))
                for(int i = 0; i < pp.numStack(); i++)
                    addPointer(processPointer(pp.task, pp.brain, pp.sl, i));
        }
        else if(pp.st == pp.numStack()){
            static vector<int> ct(tasks.size(), 0);
            if(!pp.task->sliceEnd(pp)){
                for(size_t i = 0; i < tasks.size(); i++) {
                    if(tasks[i] == pp.task){
                        ct[i]++;
                        if(ct[i] == pp.numSlice())
                            addPointer(processPointer(pp.task, pp.brain, ct[i]));
                        break;
                    }
                }
            }
        }
        else if(pp.im == -1){
            if(!pp.task->stackBegin(pp))
                addPointer(processPointer(pp.task, pp.brain, pp.sl, pp.st, -1 , -1, 0));
        }
        else if(pp.im == pp.numImage()){
            static vector<vector<int> > ct(tasks.size(), vector<int>(pp.numSlice(), 0));
            if(!pp.task->stackEnd(pp)){
                for(uint i = 0; i < tasks.size(); i++) {
                    if(tasks[i] == pp.task) {
                        ct[i][pp.sl]++;
                        if(ct[i][pp.sl] == pp.numStack())
                            addPointer(processPointer(pp.task, pp.brain, pp.sl, ct[i][pp.sl]));
                        break;
                    }
                }
            }
        }
        else if(pp.im == 0){
            processPointer ppp = pp;
            for(int i = 0; i < pp.numImage(); i++){
                //if(ppp.im > 200) {ppp.im++; continue;}
                ppp.stack()->loadImage(ppp.im);
                for(auto t : tasks){
                    ppp.task = t;
                    t->loadImage(ppp);
                }
                ppp.stack()->release(ppp.im++);
            }
            for(auto t : tasks){
                ppp.task = t;
                addPointer(ppp);
            }
        }

#pragma omp critical(ongoing)
        {
            ongoing.erase(pp);
        }
#pragma omp critical(finished)
        {
            finished.insert(pp);
        }
    }
}

void TaskManager::ioThread()
{
}

processPointer::processPointer(Task *task_, Brain *brain_, int slice_, int stack_, int step_, int part_, int image_)
{
    task = task_;
    brain = brain_;
    sl = slice_;
    st = stack_;
    step = step_;
    part = part_;
    im = image_;
}

bool processPointer::operator<(const processPointer other) const
{
    if(other.brain == nullptr) return false;
    if(brain == nullptr) return true;
    if(brain == other.brain){
        if(sl == other.sl){
            if(st == other.st){
                if(im == other.im && im != 0){
                    if(task < other.task) return true;
                }
                else if(im < other.im) return true;
            }
            else if(st < other.st) return true;
        }
        else if(sl < other.sl) return true;
    }
    else if(brain < other.brain) return true;
    return false;
}

void processPointer::print()
{
    cout << "Task\t" << task;
    pointer::print();
}
