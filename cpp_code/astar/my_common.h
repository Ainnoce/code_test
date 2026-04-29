#pragma once

namespace common{
struct Point{
    int x;
    int y;
    Point();
    Point(int x, int y);
    virtual bool operator==(const Point& p2)const;
    virtual Point& operator=(const Point& p2);
    virtual Point operator+(const Point& p2)const;
    double Manhattan_(const Point& p2)const;
    virtual ~Point() = default;
};

} // end namespace