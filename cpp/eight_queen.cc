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

/*

用i,j分别表示皇后所在的行列(或者是说i号皇后在j列),同一主对角线上的行列下标的差一样,
若用表达式i-j编号,则范围为 -n+1~n-1,所以用表达式 i-j+n对主对角线编号,范围就是1~2n-1,
同样地,负对角线上行列下标的和一样,用表达式i+j编号,则范围为 2~2n.

递归算法的回溯法是由函数调用结束自动完成的,不需要指出回溯点(类似算法2中的k=k-1),
但需要"清理现场" -- 将当前点占用的位置释放, 也就是算法 try()中的后3个赋值语句
*/

void try_queen(
    int queen_index,
    int queen_num,
    int& count,
    int* a, // 放皇后的数组
    int* col, // 列占位
    int* major_diag, // 主对角线占位
    int* minor_diag){ // 次对角线占位

    for(int j=0;j<queen_num;j++){ // 第queen_index个皇后有n个可能的位置
      if(col[j]==0&&major_diag[queen_index-j+queen_num]==0&&minor_diag[queen_index+j]==0){ // 判断位置是否冲突,若无冲突
        a[queen_index] = j;
        col[j]=1; // 占领第j列
        major_diag[queen_index-j+queen_num]=1; // 占领主对角线
        minor_diag[queen_index+j]=1; // 占领次对角线
        if(queen_index==queen_num-1){ // 已经摆完了n个皇后, 输出
            count++;
            output(a, queen_num);
        }else{ // n个皇后未摆放完, 需要继续放下一个皇后
            try_queen(queen_index+1, queen_num, count, a, col, major_diag, minor_diag);
        }
        // 回溯恢复现场
        col[j]=0;
        minor_diag[queen_index+j]=0;
        major_diag[queen_index-j+queen_num]=0;
      } // end of if, 若有冲突,换下一个位置
   }
}

void swap(int& a, int& b){
    int temp = a;
    a = b;
    b = temp;
}

void swap(int* a, int* b){
    int temp = *a;
    *a = *b;
    *b = temp;
}

/*
TODO:此算法目前还有些问题,数目不对
使用排列法解8皇后问题,
利用枚举所有1~n的排列, 从中选出满足约束条件的解来,
注意:此时,约束条件就只有"皇后不在同一对角线上",而不需要"皇后在不同列"的约束了
*/
// 输出解
void output2(int*a, int len){
  for(int i=1;i<=len;i++) {
    printf("%d ", a[i]);
  }
  printf("\n");
}
void try_permutaion(
    int queen_index,
    int queen_num,
    int& count,
    int* a, // 放皇后的数组
    int* major_diag, // 主对角线占位
    int* minor_diag){ // 次对角线占位

    if(queen_index==queen_num-1){ // 已经摆完了n个皇后, 输出
        count++;
        output(a, queen_num);
    }else{
        for(int j=queen_index;j<queen_num;j++){ // 第queen_index个皇后有n个可能的位置
          //swap(a+queen_index, a+j);
          swap(a[queen_index], a[j]); // 交换queen_index与j位置上的皇后
          if(major_diag[queen_index-a[queen_index]+queen_num]==0\
                &&minor_diag[queen_index+a[queen_index]]==0){ // 判断位置是否冲突,若无冲突

            major_diag[queen_index-a[queen_index]+queen_num]=1; // 占领主对角线
            minor_diag[queen_index+a[queen_index]]=1; // 占领次对角线
            // 尝试下一位置
            try_permutaion(queen_index+1, queen_num, count, a, major_diag, minor_diag);
            // 回溯恢复现场
            major_diag[queen_index-a[queen_index]+queen_num]=0; // 占领主对角线
            minor_diag[queen_index+a[queen_index]]=0; // 占领次对角线
          } // end of if, 若有冲突,换下一个位置
          swap(a[queen_index], a[j]);
       }
   } // end of if
}


int main(int argc, char* argv[]){
    {
       printf("\n=====1回溯法(非递归)解8皇后=======\n");
       int queen_num=8;
       int* a = new int[queen_num];
       for(int i=0;i<queen_num;i++){
         a[i]=0;
       }
       int count = backdate(queen_num, a);
       printf("queue:%d total:%d", queen_num, count);
       delete[] a;
   }

   {
       printf("\n=====2递归回溯法解8皇后=======\n");
       int queen_num=8;
       int* a = new int[queen_num];
       int* col = new int[queen_num];
       int* major_diag = new int[2*queen_num];
       int* minor_diag = new int[2*queen_num];
       for(int i=0;i<queen_num;i++){
         a[i]=0;
         col[i]=0;
         major_diag[i]=0;
         major_diag[i+queen_num]=0;
         minor_diag[i]=0;
         minor_diag[i+queen_num]=0;
       }
       int count =0;
       try_queen(0, queen_num, count, a, col, major_diag, minor_diag);
       printf("queue:%d total:%d", queen_num, count);
       delete[] a;
       delete[] col;
       delete[] major_diag;
       delete[] minor_diag;

   }

   {
       printf("\n=====3排列法解8皇后=======\n");
       int queen_num=8;
       int* a = new int[queen_num];
       int* major_diag = new int[2*queen_num];
       int* minor_diag = new int[2*queen_num];
       for(int i=0;i<queen_num;i++){
         a[i]=i; // 每个位置上放一个皇后i
         major_diag[i]=0;
         major_diag[i+queen_num]=0;
         minor_diag[i]=0;
         minor_diag[i+queen_num]=0;
       }
       int count =0;
       int start_index=0;
       try_permutaion(start_index, queen_num, count, a, major_diag, minor_diag);
       printf("queue:%d total:%d", queen_num, count);
       delete[] a;
       delete[] major_diag;
       delete[] minor_diag;
   }
}

