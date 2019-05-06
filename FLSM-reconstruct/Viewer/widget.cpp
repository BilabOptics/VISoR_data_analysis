#include "widget.h"
#include "ui_widget.h"
#include <QFileDialog>
#include <QWheelEvent>
#include <QScrollBar>
#include <QShortcut>
#include <QtMath>
#include <QTextStream>

using namespace cv;
using namespace flsm;

QPixmap MatToQPixmap(Mat img)
{
    QImage qimg((unsigned char*)img.data, img.cols, img.rows,
                 img.step, QImage::Format_Grayscale8);
    QPixmap qpix;
    qpix.convertFromImage(qimg);
    return qpix;
}

Widget::Widget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::Widget)
{
    ui->setupUi(this);
    ui->graphicsView->setScene(new QGraphicsScene());
    globalContrast = 65536;
    globalScaleLevel = 0;
    currentImageName = QString();
    setMode(FV_NULL);

    painter.scene = ui->graphicsView->scene();

    connect(ui->pbOpen, SIGNAL(clicked(bool)), this, SLOT(openFiles()));
    connect(ui->pbZoomin, SIGNAL(clicked(bool)), this, SLOT(zoomIn()));
    connect(ui->pbZoomout, SIGNAL(clicked(bool)), this, SLOT(zoomOut()));
    connect(ui->sliderContrast, SIGNAL(valueChanged(int)), this, SLOT(setContrast(int)));
    connect(ui->pbPrev, SIGNAL(clicked(bool)), this, SLOT(prevImage()));
    connect(ui->pbNext, SIGNAL(clicked(bool)), this, SLOT(nextImage()));
    connect(ui->sliderContrast, SIGNAL(valueChanged(int)), this, SLOT(refreshTextValue()));
    connect(ui->lineEdit, SIGNAL(textEdited(QString)), this, SLOT(setContrastByText()));
    connect(ui->graphicsView, SIGNAL(wheelZoomIn()), this, SLOT(zoomIn()));
    connect(ui->graphicsView, SIGNAL(wheelZoomOut()), this, SLOT(zoomOut()));
    connect(ui->spinBoxSL, SIGNAL(valueChanged(int)), this, SLOT(moveSlice(int)));
    connect(ui->spinBoxST, SIGNAL(valueChanged(int)), this, SLOT(moveStack(int)));
    connect(ui->spinBoxIM, SIGNAL(valueChanged(int)), this, SLOT(moveImage(int)));
    connect(ui->pbExport, SIGNAL(clicked(bool)), this, SLOT(exportMarks()));
    new QShortcut(QKeySequence(Qt::Key_Left), this, SLOT(prevImage()));
    new QShortcut(QKeySequence(Qt::Key_Right), this, SLOT(nextImage()));
    new QShortcut(QKeySequence(Qt::Key_Equal), this, SLOT(zoomIn()));
    new QShortcut(QKeySequence(Qt::Key_Minus), this, SLOT(zoomOut()));
}

Widget::~Widget()
{
    delete ui;
}


void Widget::bufferImage(int index)
{
    QString fn = fileNames[index];
    bufferImage(fn);
}

void Widget::bufferImage(QString file)
{
    Mat img = imageList[file];
    Mat bf(img.size(), CV_8UC1);
    convertScaleAbs(img, bf, 256.0 / globalContrast);
    if(imageBuffer.find(file) == imageBuffer.end()) {
        imageBuffer.insert(file, Mat());
        bufferedContrastValue.insert(file, globalContrast);
    }
    imageBuffer[file] = bf;
    bufferedContrastValue[file] = globalContrast;
}

void Widget::showImage()
{
    ui->graphicsView->scene()->clear();
    ui->graphicsView->scene()->addPixmap(MatToQPixmap(imageBuffer[currentImageName]));
    painter.paintMarks(currentPosition);
    ui->graphicsView->repaint();
}

void Widget::openImages(QStringList files)
{
    for(QString file : files) {
        Mat img = imread(file.toStdString(), -1);
        if(!img.empty()) {
            fileNames.append(file);
            ui->comboBox->addItem(file.section('/', -1));
            imageList.insert(file, img);
        }
    }
    if(imageList.size() > 0) {
        for(int i = 0; i < imageList.size(); i++)
            bufferImage(i);
        currentImageName = fileNames[0];
        ui->graphicsView->scene()->addPixmap(MatToQPixmap(imageBuffer[currentImageName]));
        setMode(FV_IMAGESTACK);
    }
}

void Widget::openOrigin(QStringList files)
{
    for(QString file : files) {
        if(!file.section('.', -1).compare(QString("flsm"))) {
            Slice sl(file.toStdString(), 0, 0.4875, false);
            if(!sl.initiallized) continue;
            if(brain[QString(sl.name.c_str())].name.empty())
                brain.insert(QString(sl.name.c_str()), Brain(sl.name));
            brain[QString(sl.name.c_str())].slices.push_back(sl);
        }
    }
    if(!brain.empty()) setMode(FV_ORIGIN);
    ui->lineEdit->setText("1000");
    setContrastByText();
}

void Widget::importMarks(QStringList files)
{
    QFile file(files[0]);
    file.open(QIODevice::ReadOnly);
    if(!file.isOpen()) return;
    painter.head = file.readLine();
    bool iscsv = !files[0].section('.', -1).compare(QString("csv"));
    QChar split = iscsv ? QChar(',') : QChar(' ');
    while(!file.atEnd()) {
        QString linetext(file.readLine());
        mark mk;
        QStringList textlist = linetext.split(split);
        if(textlist.size() < 4) continue;
        mk.center.x = textlist[0].toDouble();
        mk.center.y = textlist[1].toDouble();
        mk.center.z = textlist[2].toDouble();
        for(int i = 3; i < textlist.size() - 1; i++) {
            mk.features.push_back(textlist[i].toDouble());
        }
        mk.type = textlist.last().toInt();
        painter.addMark(mk);
    }
}

void Widget::setMode(ViewerMode mode)
{
    currentMode = mode;
    if(mode == FV_NULL) {
        ui->pointerWidget->setEnabled(false);
        return;
    }
    if(mode == FV_IMAGESTACK) {
        disconnect(ui->comboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(changeBrain(int)));
        connect(ui->comboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(changeImage(int)));
        ui->pointerWidget->setEnabled(false);
        changeImage(0);
        return;
    }
    if(mode == FV_ORIGIN) {
        disconnect(ui->comboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(changeImage(int)));
        connect(ui->comboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(changeBrain(int)));
        ui->pointerWidget->setEnabled(true);
        ui->comboBox->clear();
        imageList.clear();
        imageBuffer.clear();
        bufferedContrastValue.clear();
        for(Brain br : brain) {
            ui->comboBox->addItem(QString(br.name.c_str()));
        }        
        currentPosition.brain = &brain.first();
        currentPosition.sl = 0;
        currentPosition.st = 0;
        currentPosition.im = 0;
        changeBrain(0);
    }
}

void Widget::openFiles()
{
    ui->comboBox->blockSignals(true);
    QFileDialog *fileDialog = new QFileDialog(this);
    fileDialog->setWindowTitle(tr("Select Image Files"));
    fileDialog->setReadOnly(true);
    fileDialog->setFileMode(QFileDialog::ExistingFiles);
    if(fileDialog->exec() == QDialog::Accepted) {
        if(!fileDialog->selectedFiles()[0].section('.', -1).compare(QString("flsm")))
            openOrigin(fileDialog->selectedFiles());
        else if((!fileDialog->selectedFiles()[0].section('.', -1).compare(QString("marks"))) ||
                (!fileDialog->selectedFiles()[0].section('.', -1).compare(QString("csv"))))
            importMarks(fileDialog->selectedFiles());
        else openImages(fileDialog->selectedFiles());
    }
    ui->comboBox->blockSignals(false);
}

void Widget::exportMarks()
{
    ui->comboBox->blockSignals(true);
    QFileDialog *fileDialog = new QFileDialog(this);
    fileDialog->setAcceptMode(QFileDialog::AcceptSave);
    fileDialog->setWindowTitle(tr("Export marks"));
    fileDialog->setFileMode(QFileDialog::AnyFile);
    if(fileDialog->exec() != QDialog::Accepted) return;
    QFile save(fileDialog->selectedFiles()[0]);
    save.open(QIODevice::WriteOnly);
    if(!save.isOpen()) return;
    QTextStream fs(&save);
    fs << painter.head;
    for(auto m : painter.marks) {
        if(m->type == 0) continue;
        QStringList lm;
        lm.push_back(QString::number(m->center.x));
        lm.push_back(QString::number(m->center.y));
        lm.push_back(QString::number(m->center.z));
        for(auto d : m->features) lm.push_back(QString::number(d));
        lm.push_back(QString::number(m->type));
        fs << lm.join(',') << endl;
    }
    save.close();
}

void Widget::zoomIn()
{
    if(globalScaleLevel < 6) {
        ++globalScaleLevel;
        ui->graphicsView->scale(1.25, 1.25);
        ui->graphicsView->repaint();
        ui->label->setText(QString::number(qPow(1.25, globalScaleLevel) * 100) + QString("%"));
    }
}

void Widget::zoomOut()
{
    if(globalScaleLevel > -8) {
        --globalScaleLevel;
        ui->graphicsView->scale(0.8, 0.8);
        ui->graphicsView->repaint();
        ui->label->setText(QString::number(qPow(1.25, globalScaleLevel) * 100) + QString("%"));
    }
}

void Widget::setContrast(int contrastLevel)
{
    globalContrast = contrastLevel * contrastLevel;
    bufferImage(currentImageName);
    showImage();
}

void Widget::changeImage(int index)
{
    if(index < 0 || index >= imageList.size()) return;
    ui->comboBox->setCurrentIndex(index);
    currentImageName = fileNames[index];
    if(bufferedContrastValue[currentImageName] != globalContrast)
        bufferImage(index);
    showImage();
}

void Widget::changeBrain(int index)
{
    if(index < 0 || index >= brain.size()) return;
    currentPosition.brain = &brain[ui->comboBox->itemText(index)];
    setPosition(currentPosition);
    ui->spinBoxSL->setRange(0, currentPosition.numSlice() - 1);
    ui->spinBoxST->setRange(0, currentPosition.numStack() - 1);
    ui->spinBoxIM->setRange(0, 2500);//currentPosition.numImage() - 1);
}

void Widget::prevImage()
{
    switch (currentMode) {
    case 1:
        changeImage(ui->comboBox->currentIndex() - 1);
        break;
    case 2:
        moveImage(currentPosition.im - 1);
        break;
    default:
        return;
    }
}

void Widget::nextImage()
{
    switch (currentMode) {
    case 1:
        changeImage(ui->comboBox->currentIndex() + 1);
        break;
    case 2:
        moveImage(currentPosition.im + 1);
        break;
    default:
        return;
    }
}

void Widget::refreshTextValue()
{
    ui->lineEdit->setText(QString::number(globalContrast));
}

void Widget::setContrastByText()
{
    if(currentImageName.isEmpty()) return;
    globalContrast = ui->lineEdit->text().toInt();
    bufferImage(currentImageName);
    showImage();
}

void Widget::setPosition(pointer p)
{
    if(!p.isValid()) p.setValid();
    currentPosition = p;
    currentImageName = QString(p.image()->fileName.c_str());
    if(imageList.find(currentImageName) == imageList.end()) {
        if(imageList.size() >= maxBufferSize) {
            QString fn = bufferOrder.first();
            imageList.remove(fn);
            imageBuffer.remove(fn);
            p.stack()->release(p.im);
            bufferedContrastValue.remove(fn);
            bufferOrder.pop_front();
        }
        imageList.insert(currentImageName, p.stack()->getImageData(p.im));
        bufferOrder.push_back(currentImageName);
    }
    else {
        for(QList<QString>::iterator ite = bufferOrder.begin(); ite == bufferOrder.end(); ite++) {
            if(!currentImageName.compare(*ite)) {
                bufferOrder.erase(ite);
                bufferOrder.push_back(currentImageName);
            }
        }
    }
    if(bufferedContrastValue[currentImageName] != globalContrast)
        bufferImage(currentImageName);
    showImage();
    painter.target = currentPosition;
    painter.paintMarks(currentPosition);
    ui->spinBoxSL->setValue(p.sl);
    ui->spinBoxST->setValue(p.st);
    ui->spinBoxIM->setValue(p.im);
}

void Widget::moveSlice(int index)
{
    currentPosition.sl = index;
    setPosition(currentPosition);
}

void Widget::moveStack(int index)
{
    currentPosition.st = index;
    setPosition(currentPosition);
}

void Widget::moveImage(int index)
{
    currentPosition.im = index;
    setPosition(currentPosition);
}

void ImageViewer::wheelEvent(QWheelEvent *event)
{
    if(event->angleDelta().y() > 0)
        emit wheelZoomIn();
    else
        emit wheelZoomOut();
}

ImageViewer::ImageViewer(QWidget *parent)
{

}

