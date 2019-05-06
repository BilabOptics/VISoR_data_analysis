#ifndef FSTACK_H
#define FSTACK_H
#include <string>
#include <vector>
#include <unordered_set>
#include <opencv2/core.hpp>
#include <rapidjson/document.h>
#include <opencv2/videoio.hpp>

#include "fimage.h"

namespace flsm {

struct Stack
{
    //double startPosX;
    //double endPosX;
    //double posY;
    double spaceing;
    double xOffset;
    box bound;
    cv::Rect roi;

    std::string prefix;
    std::string suffix;
    std::string videoFile;
    cv::VideoCapture* video;
    std::vector<Image> images;
    std::vector<int> videoAlias;
    std::unordered_set<size_t> loadList;
    int startct;
    int endct;
    bool origin;
    bool fromVideo = false;
    bool videoInitallized = false;
    int videoPos;
    bool shift = true;
    bool allow_nulldata = true;

    Stack(int n); //test
    Stack(bool isOrigin = false, cv::Rect roi_ = cv::Rect(0, 0, 0, 0));
    Stack(double spX, std::string Prefix, std::string Suffix, int start = 0, int end = 0, bool isOrigin = false, cv::Rect roi = cv::Rect(0, 0, 0, 0));
    int loadImage();
    int loadImage(size_t idx);
    int loadImage(uint start, uint end);
    int loadImage(std::vector<size_t> ilist);
    int loadFromVideo(size_t idx);
    void release();
    void release(size_t idx);
    void release(size_t start, size_t end);
    void release(std::vector<size_t> rlist);
    int save();
    int save(uint idx);
    cv::Mat getImageData(size_t idx);
    cv::Mat getShiftImageData(size_t idx);
    cv::Point2i pictureSize();
    void append(cv::Mat img);
    void append(Image img);
    Image& operator[](size_t idx);

    std::vector<cv::Mat> genVoxelData(double scale, cv::Point2d xRange, cv::Point2d yRange, cv::Point2d zRange, double pixelSize = 1);
    void genVoxelData_(std::vector<cv::Mat>& dst, double scale, cv::Point2d xRange, cv::Point2d yRange, cv::Point2d zRange, double pixelSize);
    ~Stack();
};

//void splitSlice(Slice src, std::vector<double> splitX, std::vector<double> splitY, std::vector<Slice> dst);

}
#endif // FSTACK_H
