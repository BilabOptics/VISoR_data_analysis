#include "markpainter.h"

using namespace flsm;
using namespace cv;

markPainter::markPainter()
{
    pen = QPen(QColor(0, 0, 255));
    pen.setWidth(0);
}

void markPainter::paintMarks(pointer pos)
{
    currentPos = pos;
    QList<mark*> currentMarks = marks.values(pos);
    for(mark* mk : currentMarks) {
        Point2d p2d;
        inverseConvertPos(mk->center, pos, p2d);
        if(mk->type == mktype_true) pen.setColor(QColor(0, 255, 0));
        else if(mk->type == mktype_false) pen.setColor(QColor(255, 0, 0));
        else pen.setColor(QColor(0, 0, 255));
        markItem* mkItem = new markItem(p2d.x - 14, p2d.y - 14, 28, 28, pen);
        mkItem->mk = mk;
        scene->addItem(mkItem);
    }
}

void markPainter::addMark(mark mk)
{
    pointer pos = target;
    for(pos.st = 0; pos.st < target.numStack(); ++pos.st) {
        Stack* st = pos.stack();
        if(mk.center.y > st->bound.ye) continue;
        if(mk.center.y > st->bound.ys) {
            Point2d p2d;
            inverseConvertPos(mk.center, pos, p2d);
            mark* m = new mark();
            (*m) = mk;
            marks.insert(pos, m);
        }
        else break;
    }
}

void markItem::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
    if(mk->type == mktype_true) mk->type = mktype_false;
    else if(mk->type == mktype_false) mk->type = mktype_default;
    else mk->type = mktype_true;
    reset();
}

markItem::markItem(qreal x, qreal y, qreal w, qreal h, QPen pen) : QGraphicsEllipseItem(x, y, w, h)
{
    setPen(pen);
}

void markItem::reset()
{
    QPen pen_ = pen();
    pen_.setWidth(0);
    if(mk->type == mktype_true) pen_.setColor(QColor(0, 255, 0));
    else if(mk->type == mktype_false) pen_.setColor(QColor(255, 0, 0));
    else pen_.setColor(QColor(0, 0, 255));
    setPen(pen_);
}
