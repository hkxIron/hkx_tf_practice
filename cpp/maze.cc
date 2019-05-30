/*
g++ -std=c++11 -o maze maze.cc &&./maze

*/

#include <cstdio>
#include <stdlib.h>
#include <iostream>
#include <cstring>
#include<algorithm>
using namespace std;

const int N = 8;
const int D = 2; // 死胡同,如果4个方向搜索完毕,仍未找到出口,设置为死胡同
const int S = 3; // 走过的标志
int maze[N][N] = { // 1代表墙壁,0代表可走
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 1, 1, 1, 1, 0, 1, 0,
    0, 0, 0, 0, 1, 0, 1, 0,
    0, 1, 0, 0, 0, 0, 1, 0,
    0, 1, 0, 1, 1, 0, 1, 0,
    0, 1, 0, 0, 0, 0, 1, 1,
    0, 1, 0, 0, 1, 0, 0, 0,
    0, 1, 1, 1, 1, 1, 1, 0,
};

int maze_bak[N][N]; // 1代表墙壁,0代表可走

struct Direction{
    int dx;
    int dy;
};

Direction direction[] ={
    {1,0}, // 右
    {-1,0}, // 左
    {0,-1}, // 上
    {0,1},// 下
};

bool check(int x, int y){
   if(x<0||x>N-1||y<0||y>N-1||maze[y][x]!=0)
     return false;
   else
     return true;
}

void output(){
    int path_length=0;
    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
           if(maze[i][j]==S){
             printf(". ");
             path_length++;
           } else {
             printf("%d ", maze[i][j]);
           }
        }
        printf("\n");
    }
    printf("path length:%d\n", path_length);
}

/*
深度优先搜索
*/
void search(int x, int y, int& step_count){
    // 4个方向搜索
    int newX,newY;
    for(int k=0;k<4;k++){
       Direction dt = direction[k];
       newX=x+dt.dx;
       newY=y+dt.dy;
       if(check(newX, newY)) {// 新位置可以用
           maze[newY][newX]=S;
           step_count++;
           if(newY==N-1&&newX==N-1) {
                output();
           }else{
                //搜索新位置
                search(newX, newY, step_count);
           }
       }
    }
    // 4个方向都不行,死胡同
    maze[y][x]=D;
}

struct Step{
    int x;
    int y;
    int pre;
};

Step queue[N*N+1];// 队列

void output_width(int q_end){
    printf("(%d,%d)", queue[q_end].x, queue[q_end].y);
    maze[queue[q_end].y][queue[q_end].x]=S;
    while(queue[q_end].pre!=-1){
       q_end=queue[q_end].pre;
       printf("<-(%d,%d)", queue[q_end].x, queue[q_end].y);
       maze[queue[q_end].y][queue[q_end].x]=S;
    }
    printf("\n");
    output();
}

/*
广度优先搜索,使用队列进行搜索,也称为分支界限法

广度优先搜索,是按层次遍历(所有层的权重相同),因此可以找到迷宫的最优解.
*/
void search_width(){
    maze[0][0] = S;// 置第1个元素为已访问
    queue[0].pre=-1;
    queue[0].x=0;
    queue[0].y=0;
    int q_head=-1;
    int q_end=0;// 初始化队列中有一个元素

    // 队列不空,则需要继续搜索
    while(q_head!=q_end){
        q_head++; // 出队
        // 4个方向搜索
        int newX,newY;
        Step head=queue[q_head];
        for(int k=0;k<4;k++){
           Direction dt = direction[k];
           newX=head.x+dt.dx;
           newY=head.y+dt.dy;
           if(check(newX, newY)) {// 新位置可以用
               q_end++; // 入队
               queue[q_end].x=newX;
               queue[q_end].y=newY;
               queue[q_end].pre=q_head;
               maze[newY][newX]=S;
               if(newY==N-1&&newX==N-1) {
                    output_width(q_end);
                    return;// 找到一个就返回
               }
           }
        }
        // 4个方向都不行,死胡同
        maze[head.y][head.x]=D;
    }
}

void reset(){
    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
            maze[i][j]=maze_bak[i][j];
        }
    }
}

int main(int argc, char* argv[]){

    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
            maze_bak[i][j]=maze[i][j];
        }
    }

    {
        printf("\n原迷宫为:\n");
        output();
        printf("\n深度优先搜索的一种走法:\n");
        maze[0][0] = S;// 入口坐标设置为已走
        int step_count=1;
        search(0,0, step_count);
        printf("step count:%d", step_count);
    }

    {
        reset();
        printf("\n原迷宫为:\n");
        output();
        printf("\n广度优先搜索的一种走法:\n");
        search_width();
    }
    return 0;
}

