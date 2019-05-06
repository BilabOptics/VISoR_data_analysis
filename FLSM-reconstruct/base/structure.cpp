#include "structure.h"
#include <cmath>

using namespace flsm;
using namespace std;
using namespace cv;

Structure::Structure(Segment first)
{
    segments.push_back(first);
    position = first.position;
}

double Structure::match(Segment next)
{
    Segment prev = segments[segments.size() - 1];
    double shift = position.stack()->spaceing / position.slice()->pixelSize / sqrt(2);
    Point2d cp(prev.center.x, prev.center.y + shift);
    Point2d cn(next.center.x, next.center.y);
    return 1 - 0.2 * dist(cp, cn);
}

void Structure::append(Segment next)
{
    segments.push_back(next);
}

double Structure::correctness()
{
    double cor = 0;
    double its = 0;
    for(Segment seg : segments) {
        double weight = seg.intensity * seg.area * seg.confidence();
        its += weight;
        cor += seg.correctness() * weight;
    }
    cor /= its;
    double rec = 0;
    its = 0;
    double shift_ = position.stack()->spaceing / position.slice()->pixelSize / sqrt(2);
    for(uint i = 1; i < segments.size(); i++) {
        Point2d cp(segments[i - 1].center.x, segments[i - 1].center.y + shift_);
        Point2d cn(segments[i].center.x, segments[i].center.y);
        double weight = segments[i].intensity * segments[i].area * segments[i].confidence()
                * segments[i - 1].intensity * segments[i - 1].area * segments[i - 1].confidence();
        its += weight;
        rec += (1 - dist(cp, cn) / segments[i].radius / 2) * weight * 0.8;
    }
    cor += rec / its;
    its = intensity();
    for(auto seg : segments)
        if(its == seg.intensity) {
            cor += fmin(sqrt(seg.area) / seg.radius * 0.564 - 0.6, 1.5 - sqrt(seg.area) / seg.radius * 0.564) * 1.2;
            break;
        }
    return cor;
}

Point3d Structure::center()
{
    double its = 0;
    Point3d sp;
    for(auto seg : segments) {
        if(seg.intensity > its) {
            its = seg.intensity;
            sp = convertPos(seg.center, seg.position);
        }
    }
    return sp;
}

double Structure::intensity()
{
    double its = 0;
    for(auto seg : segments)
        if(its < seg.intensity)
            its = seg.intensity;
    return its;
}

double Structure::totalIntensity()
{
    double its = 0;
    for(auto seg : segments) {
        its += seg.intensity * seg.area;
    }
    return its;
}

double Structure::shift(int i, int j)
{
    double shift_ = position.stack()->spaceing / position.slice()->pixelSize / sqrt(2);
    Point2d cp(segments[i].center.x, segments[i].center.y);
    Point2d cn(segments[j].center.x, segments[j].center.y + shift_ * (i - j));
    return dist(cp, cn);
}

std::vector<int> Structure::peaks()
{
    vector<int> pk;
    if(segments.size() < 3) return pk;
    for(int i = 1; i < segments.size() - 1; ++i) {
        if(segments[i].intensity > segments[i - 1].intensity
                && segments[i].intensity > segments[i + 1].intensity)
            pk.push_back(i);
    }
    return pk;
}

std::vector<int> Structure::pits()
{
    vector<int> pk;
    if(segments.size() < 3) return pk;
    for(int i = 1; i < segments.size() - 1; ++i) {
        if(segments[i].intensity < segments[i - 1].intensity
                && segments[i].intensity < segments[i + 1].intensity)
            pk.push_back(i);
    }
    return pk;
}

int Structure::peak()
{
    double its = 0;
    int p = -1;
    for(int i = 0; i < segments.size(); ++i)
        if(its < segments[i].intensity) {
            its = segments[i].intensity;
            p = i;
        }
    return p;
}

Segment &Structure::last()
{
    return segments[segments.size() - 1];
}

StructureMark::StructureMark(Structure src)
{
    correctness = src.correctness();
    intensity = src.intensity();
    center = src.center();
}
