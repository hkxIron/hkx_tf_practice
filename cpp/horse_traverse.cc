/*
g++ -std=c++11 -o horse_traverse horse_traverse.cc &&./horse_traverse

*/

#include <cstdio>
#include <stdlib.h>
#include <iostream>
#include <cstring>
#include<algorithm>
#include<math.h>
using namespace std;

/*
马的遍历:
在n*m的棋盘中, 马只能走"日"字, 马从位置(x,y)处出发,把棋盘中每个点都走一次,
且只走一次, 找出所有路径.


分析:
马在棋盘上的点上行走,所以这里的棋盘是指行有n条边,列有m条边的,而马在不出边界的情况下,
有8个方向可以行走(走日字),如当前坐标为(x,y)则行走后的坐标为:
(x+1,y+2)
(x+1,y-2)
(x+2,y+1)
(x+2,y-1)
(x-1,y-2)
(x-1,y+2)
(x-2,y-1)
(x-2,y+1)

搜索过程是从任一点出发,按深度优先原则,从8个方向中尝试一个可以走的点,
直到走过棋盘上所有n*m个点,用递归算法易实现此过程.
*/

const int N=5;

struct Move{
   int dx;
   int dy;
};

bool check_valid(int a[][N], int x, int y, int row, int col){
    if(x>=0&&y>=0&&x<col&&y<row&&a[y][x]==-1) return true;
    else return false;
}

bool output(int a[][N], int row, int col){
    for(int i=0;i<row;i++){
        for(int j=0;j<col;j++){
            if(a[i][j]==-1){
                printf("* ");
            }else{
                printf("%d ", a[i][j]);
            }
        }
        printf("\n");
    }
    printf("\n");
}

void search(int a[][N],
            int& count,
            int x,
            int y,
            const int row,
            const int col,
            int direction_num,
            int step,
            Move* move){

    for(int i=0;i<direction_num;i++){
        int cur_x = x+ move[i].dx;
        int cur_y = y+ move[i].dy;
        if(check_valid(a, cur_x, cur_y, row, col)){
           //printf("cur_x:%d cur_y:%d\n", cur_x, cur_y);
           a[cur_y][cur_x] = step; // 占领当前位置
           if(step == row*col){
              // 走法加1
              count++;
              output(a, row, col);
           } else {
              // step+1
              search(a, count, cur_x, cur_y, row,
                     col, direction_num, step+1, move);
           }
           // --------
           a[cur_y][cur_x] = -1; // 回溯,恢复未走标志
        }// end of if
    }// end of for
}

/*
 素数环问题:
 把从1到20这20个数摆成一个环, 要求相邻的两个数的和是一个素数

 非常明显,此题是需要进行尝试并且回溯,从1开始,
 每个位置有2~20共19种可能, 约束条件就是填进去的数满足以下两条
(1) 与前面的数不相同
(2)与前面相邻数据的和是一个素数
*/

class Prime{
    public:
        int N;
        int* a;
        int count;
        Prime(int n):N(n),count(0){
            a = new int[N];
            for(int i=0;i<N;i++) {
                a[i] = 0;
            }
            a[0] = 1;
        }

        ~Prime(){
            if(a!=NULL){
                delete[] a;
            }
        }

        // 判断当前num是否与前面 (<position)的位置元素有重复
        bool check_no_duplicated(int num, int position){
            for(int i=0;i<position;i++){
                if(a[i]==num) return false;
            }
            return true;
        }

        bool check_prime(int num) {
            int limit = sqrt(num);
            for(int i=2;i<=limit;i++){
                if(num%i==0)  return false;
            }
            return true;
        }

        // 判断相邻的两个元素之和是否为素数
        bool check_all_prime(int num, int position){
            if(position==N-1) { // 最后一个元素需要检查与第1个元素之和
               return check_prime(num+a[position-1])&&check_prime(num+a[0]);
            }else{
               return check_prime(num+a[position-1]);
            }
        }

        bool output(){
            for(int i=0;i<N;i++){
                printf("%d ", a[i]);
            }
            printf("\n");
        }

        void try_prime(int position){
            for(int num=2; num<= N; num++){
               if (check_no_duplicated(num, position)&&check_all_prime(num, position)) {
                    a[position] = num;
                    if(position==N-1){
                        this->count++;
                        this->output();
                    }else{
                        try_prime(position+1);
                    }
                    // 恢复现场
                    a[position] = 0;
               }
            }
        }
};

int main(int argc, char* argv[]){
   {
        const int row = N;
        const int col = N;
        int a[row][col];
        for(int i=0;i<row;i++){
            for(int j=0;j<col;j++){
                a[i][j]=-1;
            }
        }
        Move move[] = {
                   {1,2},
                   {1,-2},
                   {2,1},
                   {2,-1},
                   {-1,-2},
                   {-1,2},
                   {-2,-1},
                   {-2,1},
        };
        int count=0;
        int step=1;
        a[0][0] = step; // 起始点为(0,0)
        int direction_num = sizeof(move)/sizeof(Move);
        search(a, count, 0, 0, row, col, direction_num, step+1, move);
        printf("direction_num:%d count:%d\n", direction_num, count);
   }

   {
        printf("\n素数环问题\n");
        Prime prime = Prime(8);
        prime.try_prime(1);
        printf("N:%d count:%d\n", prime.N, prime.count);
   }
}


