#pragma once

#include<iostream>
#include<vector>
#include<list>
#include<memory>
#include"my_common.h"

using std::vector;
using std::list;
using std::shared_ptr;



namespace planning{

const double k_cost_adjacent = 1.0 ;
const double k_cost_corner = 2.0 ;

class AstarNode{
public:
    AstarNode(common::Point& p, double map_cost = 0);

    common::Point point_;
    double F_;
    double G_;
    double H_;
    double map_cost_;
    shared_ptr<AstarNode> parent_;
    double CalG_(const shared_ptr<AstarNode> lastNode)const;
    double CalCost_(const common::Point& endpoint);
};

class Astar{
public:
    explicit Astar(vector<vector<double>> map);
    list<shared_ptr<AstarNode>> GetPath_(common::Point& startPoint, 
                            common::Point& endPoint, bool isIgnoreObs = false);
private:
    bool IsCanReach_(const common::Point& p, bool isIgnoreObs = false)const;
    shared_ptr<AstarNode> FindPath_(common::Point& startPoint, 
                            common::Point& endPoint, bool isIgnoreObs = false);
    shared_ptr<AstarNode> GetLeastFNode_()const;
    shared_ptr<AstarNode> IsInList_(const common::Point& p, list<shared_ptr<AstarNode>> _list) const;


private:
    vector<vector<double>> cost_map_;
    list<shared_ptr<AstarNode>> openlist_;
    list<shared_ptr<AstarNode>> closelist_;
};

bool InPath(int row, int col, const list<std::shared_ptr<AstarNode>>& path);

} // namespace planning