#include"my_common.h"

namespace common{

Point::Point():x(0), y(0){}

Point::Point(int x, int y):x(x), y(y){}

bool Point::operator==(const Point& p2)const{
    return (x==p2.x && y==p2.y);
}

Point& Point::operator=(const Point& p2){
    if(!(*this == p2)){
        x = p2.x;
        y = p2.y;
    }
    return *this;
}

Point Point::operator+(const Point& p2)const{
    return Point(x + p2.x, y + p2.y);
}

double Point::Manhattan_(const Point& p2)const{
    return (x-p2.x)*(x-p2.x) + (y-p2.y)*(y-p2.y);
}

} // end common