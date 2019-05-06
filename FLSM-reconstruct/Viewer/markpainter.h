#ifndef MARKSPAINTER_H
#define MARKSPAINTER_H
#include <QPainter>
#include <QMap>
#include <QGraphicsScene>
#include <QGraphicsEllipseItem>
#include <mark.h>

namespace flsm {

class markItem : public QGraphicsEllipseItem
{
public:
    mark* mk;
    void mousePressEvent(QGraphicsSceneMouseEvent* event);
    markItem(qreal x, qreal y, qreal w, qreal h, QPen pen);
    void reset();
};

class markPainter
{
public:
    markPainter();
    QPen pen;
    QBrush brush;
    QGraphicsScene* scene;
    QMultiMap<pointer, mark*> marks;
    QString head;
    pointer currentPos;
    void paintMarks(pointer pos);
    void addMark(mark mk);
    pointer target;
};

}

#endif // MARKSPAINTER_H
