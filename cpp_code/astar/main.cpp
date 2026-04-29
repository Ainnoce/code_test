    #include<iostream>
    #include<vector>
    #include<unordered_map>
    #include"a_star.h"

int main() {


   std::cout << "astar" << std::endl;
   //初始化地图;用二维矩阵代表地图;1表示障碍物;0表示可通
   vector<vector<double>> map{ {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                               {-1,  0,  0, -1, -1,  0, -1,  0,  0,  0,  0, -1},
                               {-1,  0,  0, -1, -1,  0,  0,  0,  0,  0,  0, -1},
                               {-1,  0,  0,  0,  0,  0, -1,  0,  0, -1, -1, -1},
                               {-1, -1, -1,  0,  0,  0,  0,  0, -1, -1,  0, -1},
                               {-1, -1,  0, -1,  0,  0,  0,  0,  0,  0,  0, -1},
                               {-1,  0, -1,  0,  0,  0,  0,  1,  0,  0,  0, -1},
                               {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1} };
   planning::Astar astar(map);

   //设置起始和结束点
   common::Point start(1, 1);
   common::Point end(6, 10);

   // A*算法找寻路径
   list<std::shared_ptr<planning::AstarNode>>path = astar.GetPath_(start, end, true);

   // 打印结果
   for (auto& p : path) {
       std::cout << "(" << p->point_.x << "," << p->point_.y << ") ";
   }
   std::cout << "\n";

    for (int row = 0; row < map.size(); ++row) {
        for (int col = 0; col < map[0].size(); ++col) {
            if (planning::InPath(row, col, path)) {
                if (map[row][col] == -1) {
                    std::cout << " e ";
                }
                else {
                    std::cout << " * ";
                }
            }
            else if(map[row][col] == 0){
                std::cout << " 0 ";
            }
            else {
                std::cout << map[row][col] << " ";
            }
        }
        std::cout << "\n";
    }
   return 0;
} // end main