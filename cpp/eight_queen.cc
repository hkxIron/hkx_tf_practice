/*
g++ -std=c++11 -o eight_queen eight_queen.cc &&./eight_queen

*/

#include <cstdio>
#include <stdlib.h>
#include <iostream>
#include <cstring>
#include<algorithm>
using namespace std;

/**
八皇后问题:
在8*8的棋盘中,放置8个皇后,任意两个皇后不能在同一行,
同一列,同一对角线(包括正对角线以及负对角线),求所有的解

xi,xj表示 两个皇后i,j的位置,则有:
1. 不在同一列的表达式: xi != xj
2. 不在同一主对角线: xi-i != xj-j => xi - xj != i-j
3. 不在同一负对解线: xi+i != xj+j => xi - xj != j-i

因此条件2,3可以合并为: |xi-xj| != |i-j|

回溯法:
数组+循环控制+回溯 实现任意n层循环的嵌套的功能,本质上是对隐式图的 "深度优先搜索算法"

8*8: 92个解
10*10: 724个解
*/
int abs(int x){
    if(x>=0) return x;
    else return -x;
}

// 判断k与前面的皇后是否冲突
bool check_valid(int*a, int k){
  for(int i=0;i<k;i++) {
    if(abs(a[i]-a[k])==abs(i-k)|| a[i]==a[k]){
        return false;//至少有一个冲突
    }
  }
  return true;// 无冲突
}

// 输出解
void output(int*a, int len){
  for(int i=0;i<len;i++) {
    printf("%d ", a[i]);
  }
  printf("\n");
}

int backdate(int queen_num, int* a) {
    int count=0;
    int k=0 ;
    a[k]=-1;  // 从第0个位置开始
    while(k>-1){
       a[k]+=1; // 尝试所有位置
       // 为第k个皇后搜索位置
       while(a[k]<queen_num&&check_valid(a, k)==false) {
          a[k]+=1;
       }
       if(a[k]<queen_num){
           if(k==queen_num-1) {// 找到一组解
             count+=1;
             output(a, queen_num);
           }else{
             k+=1; // 前k个皇后找到位置, 继续为第k+1个皇后找位置
             a[k]=-1; // 下个皇后一定要从头开始搜索
           }
       }else{ // 第k个皇后找不到合适的位置, 回溯重填第k-1个皇后
            k-=1; // 回溯
       }
    }
    return count;
}

int main(int argc, char* argv[]){
   int queen_num=8;
   int* a = new int[queen_num];
   for(int i=0;i<queen_num;i++){
     a[i]=0;
   }
   int count = backdate(queen_num, a);
   printf("queue:%d total:%d", queen_num, count);
   delete[] a;
}

