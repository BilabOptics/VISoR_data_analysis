#ifndef IMAGE_H
#define IMAGE_H
#include <opencv2/core.hpp>
#include <misc.h>

namespace flsm {

struct Image
{
    double posX;
    double posY;
    //double imageHeight;

    std::string fileName;
    cv::Mat data;
    cv::Rect roi;
    int flipcode = 0;
    int loadError = 0;

    Image(std::string file = std::string(), cv::Mat imgData = cv::Mat(), cv::Rect roi_= cv::Rect(0, 0, 0, 0));
    bool load(bool isOrigin = false);
    int save();
    void release(bool isOrigin = false);
    bool isLoaded();
    ~Image();
};

//cv::Mat flsmImread(const std::string &filename,int flags=-1);

}

#endif // IMAGE_H
