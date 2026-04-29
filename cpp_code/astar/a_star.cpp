#include "a_star.h"
#include <math.h>


namespace planning{
vector<common::Point> surroundBias = 
                {   common::Point(-1, -1), common::Point(0, -1),common::Point(1, -1),
                    common::Point(-1,  0),                      common::Point(1,  0),
                    common::Point(-1,  1), common::Point(0,  1),common::Point(1,  1)};

    // AstarNode
    AstarNode::AstarNode(common::Point& p, double map_cost):point_(p), map_cost_(map_cost),
                        F_(0), G_(0), H_(0), parent_(nullptr){
    }   

    double AstarNode::CalG_(const shared_ptr<AstarNode> lastNode)const{
        if(lastNode == nullptr){
            return 0;
        }else{
            double extraG = abs(lastNode->point_.x - point_.x) + abs(lastNode->point_.y - point_.y)==1 
                        ? k_cost_adjacent : k_cost_corner;
            return lastNode->G_ + extraG;
        }
    }

    double AstarNode::CalCost_(const common::Point& endPoint){
        G_ = CalG_(parent_);
        H_ = point_.Manhattan_(endPoint);
        F_ = G_ + H_ + map_cost_;

        return F_;
    }

    // Astar
    Astar::Astar(vector<vector<double>> map){
        cost_map_ = map;
    }

    bool Astar::IsCanReach_(const common::Point& p, bool isIgnoreObs)const{
        int x = p.x;
        int y = p.y;
        if(x < 0 || y < 0 || x >= cost_map_.size() || y >= cost_map_[0].size()
            || (cost_map_[x][y] == -1 && !isIgnoreObs)){
            return false;
        }
        return true;
    }

    shared_ptr<AstarNode> Astar::GetLeastFNode_()const{
        shared_ptr<AstarNode> leastFNode = openlist_.front();
        for(auto n:openlist_){
            if(n->F_ < leastFNode->F_){
                leastFNode = n;
            }
        }
        return leastFNode;
    }

    shared_ptr<AstarNode> Astar::IsInList_(const common::Point& p, list<shared_ptr<AstarNode>> _list) const {
        for(auto n:_list){
            if(p == n->point_){
                return n;
            }
        }
        return nullptr;
    }

    shared_ptr<AstarNode> Astar::FindPath_(common::Point& startPoint, 
                            common::Point& endPoint, bool isIgnoreObs){
        shared_ptr<AstarNode> node = 
            std::make_shared<AstarNode>(startPoint, cost_map_[startPoint.x][startPoint.y]);
        openlist_.push_back(node);
        
        while(!openlist_.empty()){
            shared_ptr<AstarNode> curNode = GetLeastFNode_();
            openlist_.remove(curNode);
            closelist_.push_back(curNode);
                    
            for(auto p:surroundBias){
                // search new points
                common::Point surroundPoint = curNode->point_ + p;
                shared_ptr<AstarNode> surroundNode = std::make_shared<AstarNode>(surroundPoint);

                // arrived end point
                if(surroundPoint == endPoint){
                    surroundNode->parent_ = curNode;
                    return surroundNode;
                }

                // valid point
                if(IsCanReach_(surroundPoint) && IsInList_(surroundPoint, closelist_) == nullptr){
                    shared_ptr<AstarNode> node = IsInList_(surroundPoint, openlist_);
                    if(node != nullptr){ // already in openlist_
                        if(surroundNode->CalG_(curNode) < node->G_){
                            node->parent_ = curNode;
                            double g = node->CalCost_(endPoint);
                        }
                    }else{ // new Node
                        surroundNode->parent_ = curNode;
                        surroundNode->CalCost_(endPoint);
                        openlist_.push_back(surroundNode);
                    }
                } // end if IsCanReach_
            } // end for
        } // end while

        return nullptr;
         
    } // end FindPath

    list<shared_ptr<AstarNode>> Astar::GetPath_(common::Point& startPoint, 
                            common::Point& endPoint, bool isIgnoreObs){
        list<shared_ptr<AstarNode>> path;
        if(IsCanReach_(startPoint) && IsCanReach_(endPoint)){
            shared_ptr<AstarNode> resNode = FindPath_(startPoint, endPoint, isIgnoreObs);
            while(resNode != nullptr){
                path.push_front(resNode);
                resNode = resNode->parent_;
            }
        }else{
            std::cout << "invalid start or end point..." << std::endl;
        }
        return path;
    }

    bool InPath(int row, int col, const list<std::shared_ptr<AstarNode>>& path) {
        for (const auto p : path) {
            if (row == p->point_.x && col == p->point_.y) {
                return true;
            }
        }
        return false;
    }


} // end namespace planning



// int main() {


//    std::cout << "astar" << std::endl;
//    //初始化地图;用二维矩阵代表地图;1表示障碍物;0表示可通
//    vector<vector<double>> map{ {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
//                                {-1,  0,  0, -1, -1,  0, -1,  0,  0,  0,  0, -1},
//                                {-1,  0,  0, -1, -1,  0,  0,  0,  0,  0,  0, -1},
//                                {-1,  0,  0,  0,  0,  0, -1,  0,  0, -1, -1, -1},
//                                {-1, -1, -1,  0,  0,  0,  0,  0, -1, -1,  0, -1},
//                                {-1, -1,  0, -1,  0,  0,  0,  0,  0,  0,  0, -1},
//                                {-1,  0, -1,  0,  0,  0,  0,  1,  0,  0,  0, -1},
//                                {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1} };
//    planning::Astar astar(map);

//    //设置起始和结束点
//    common::Point start(1, 1);
//    common::Point end(6, 10);

//    // A*算法找寻路径
//    list<std::shared_ptr<planning::AstarNode>>path = astar.GetPath_(start, end, true);

//    // 打印结果
//    for (auto& p : path) {
//        std::cout << "(" << p->point_.x << "," << p->point_.y << ") ";
//    }
//    std::cout << "\n";

//     for (int row = 0; row < map.size(); ++row) {
//         for (int col = 0; col < map[0].size(); ++col) {
//             if (planning::InPath(row, col, path)) {
//                 if (map[row][col] == -1) {
//                     std::cout << " e ";
//                 }
//                 else {
//                     std::cout << " * ";
//                 }
//             }
//             else if(map[row][col] == 0){
//                 std::cout << " 0 ";
//             }
//             else {
//                 std::cout << map[row][col] << " ";
//             }
//         }
//         std::cout << "\n";
//     }
//    return 0;
// } // end main