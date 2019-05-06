#ifndef MISC_H
#define MISC_H

#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace flsm {

double dist(cv::Point2i, cv::Point2i);
double dist(cv::Point2d, cv::Point2d);

std::vector<std::vector<cv::Point2i> > distSearch(std::vector<cv::Point2i> src, double maxdist, double mindist);

bool testRun(int min, int max);

struct box
{
    double xs = 0;
    double xe = 1;
    double ys = 0;
    double ye = 1;
    double zs = 0;
    double ze = 1;
    double xl() { return xe - xs;}
    double yl() { return ye - ys;}
    double zl() { return ze - zs;}
    inline box operator&&(const box other);
    box();
};

std::string quot(std::string str);

std::vector<std::string> getFoldersInFolder(std::string folder);
std::vector<std::string> getFilesInFolder(std::string folder);
int createFolder(std::string folder);

}

#endif // MISC_H
