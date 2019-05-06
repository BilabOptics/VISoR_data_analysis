#include "misc.h"
#include <cmath>
#include <future>
#include <chrono>
#include <random>
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>

using namespace flsm;
using namespace std;
using namespace cv;

double flsm::dist(Point2i a, Point2i b)
{
    return sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}

double flsm::dist(Point2d a, Point2d b)
{
    return sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}

bool flsm::testRun(int min, int max)
{/*
    long seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::minstd_rand0 generator (seed);
    int t = round(min + (max - min) * (double(generator()) / generator.max()));
    this_thread::sleep_for(chrono::milliseconds(t));*/
    return false;
}

string flsm::quot(string str)
{
    return string("\"") + str + "\"";
}

vector<string> flsm::getFoldersInFolder(string folder)
{
    DIR* dir;
    dirent *ent;
    dir = opendir(folder.c_str());
    vector<string> folderlist;
    if(dir == NULL) return folderlist;
    while((ent = readdir (dir)) != NULL)
        //if(int(ent->d_type) == DT_DIR)
        if(ent->d_name[0] != '.')
            folderlist.push_back(string(ent->d_name));
    closedir (dir);
    return folderlist;
}


vector<string> flsm::getFilesInFolder(string folder)
{
    DIR* dir;
    dirent *ent;
    dir = opendir(folder.c_str());
    vector<string> filelist;
    if(dir == NULL) return filelist;
    while((ent = readdir (dir)) != NULL)
        //if(int(ent->d_type) == DT_REG)
            filelist.push_back(string(ent->d_name));
    closedir (dir);
    return filelist;
}

int flsm::createFolder(string folder)
{
#ifdef _WIN32
    string cmd = string("md \"") + folder + "\"";
    //cout << cmd << endl;
    return system(cmd.c_str());
#else
    //mode_t nMode = 0733;
    //return mkdir(folder.c_str(), nMode);
    string cmd = string("mkdir -p \"") + folder + "\"";
    return system(cmd.c_str());
#endif
}

vector<vector<Point2i> > flsm::distSearch(std::vector<Point2i> src, double maxdist, double mindist)
{
    vector<vector<Point2i> > dst(src.size(), vector<Point2i>());
    if(src.empty()) return dst;
    for(uint i = 0; i < src.size() - 1; ++i) {
        Point2i p = src[i];
        for(uint j = i + 1; j < src.size(); ++j) {
            Point2i q = src[j];
            double d = dist(p, q);
            if(d < maxdist && d > mindist) {
                dst[i].push_back(Point2i(q.x - p.x, q.y - p.y));
                dst[j].push_back(Point2i(p.x - q.x, p.y - q.y));
            }
        }
    }
    return dst;
}


box::box()
{

}
