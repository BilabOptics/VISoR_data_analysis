#ifndef WIDGET_H
#define WIDGET_H

#include <QWidget>
#include <QGraphicsView>
#include <QMap>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "fbrain.h"
#include "markpainter.h"

const int maxBufferSize = 20;

namespace Ui {
class Widget;
}

enum ViewerMode {
    FV_NULL = 0,
    FV_IMAGESTACK = 1,
    FV_ORIGIN = 2,
    FV_RESULT = 3
};

class ImageViewer : public QGraphicsView
{
    Q_OBJECT
    void wheelEvent(QWheelEvent* event);
public:
    explicit ImageViewer(QWidget *parent = 0);

signals:
    void wheelZoomIn();
    void wheelZoomOut();
};

class Widget : public QWidget
{
    Q_OBJECT

public:
    explicit Widget(QWidget *parent = 0);
    ~Widget();
    void bufferImage(int index);
    void bufferImage(QString file);
    void showImage();
    void openImages(QStringList files);
    void openOrigin(QStringList files);
    void importMarks(QStringList files);
    void setMode(ViewerMode mode);

    ViewerMode currentMode = FV_NULL;
    QStringList fileNames;
    QList<QString> bufferOrder;
    QMap<QString, cv::Mat> imageList;
    QMap<QString, cv::Mat> imageBuffer;
    QMap<QString, int> bufferedContrastValue;
    QPixmap currentImage;
    QString currentImageName;
    int globalContrast;
    int globalScaleLevel;

    QMap<QString, flsm::Brain> brain;
    flsm::pointer currentPosition;
    flsm::markPainter painter = flsm::markPainter();

private slots:
    void openFiles();
    void exportMarks();
    void zoomIn();
    void zoomOut();
    void setContrast(int contrastLevel);
    void changeImage(int index);
    void changeBrain(int index);
    void prevImage();
    void nextImage();
    void refreshTextValue();
    void setContrastByText();
    void setPosition(flsm::pointer p);
    void moveSlice(int index);
    void moveStack(int index);
    void moveImage(int index);

private:
    Ui::Widget *ui;
};

#endif // WIDGET_H
