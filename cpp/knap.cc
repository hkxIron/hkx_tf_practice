/*
g++ -std=c++11 -o knap knap.cc &&./knap
*/

#include <cstdio>
#include <stdlib.h>
#include <iostream>
#include <cstring>
#include<algorithm>
using namespace std;

/**
0/1背包
 给定一个背包容量为S, 有n件物品,重量分别为w1,w2,...,wn,能否从中选取若干物品使得重量为恰好为S
*/
// 时间复杂度为:O(2^n)
bool knap1(int remain_sum, int* weight, int good_index){
    if(remain_sum==0){
        printf("\n");
        return true;
    }
    else if(remain_sum<0||(remain_sum>0&&good_index<0)){
        return false;
    }else{
        // 选择第i 个物品
        if(knap1(remain_sum-weight[good_index], weight, good_index-1)==true){
            printf("%d ", weight[good_index]);
            return true;
        }else{
            // 不选择第i个物品
            return knap1(remain_sum, weight, good_index-1);
        }
    }
}

/*
0/1背包
给定一个背包容量为S, 有n件物品,重量分别为w1,w2,...,wn, 利润分别为:p1,p2,...,pn,
从中选取若干物品使得总利润最大,返回最大利润
时间复杂度:O(2^n)
*/
inline int max(int a, int b){
   return a>=b?a:b;
}
int knap2(int remain_sum, int* weight, int* profit, int good_index){
   if(good_index<0||remain_sum<=0){
     printf("\n");
     return 0;
   }
   // 不选当前物品
   int max_no_choose = knap2(remain_sum, weight, profit, good_index-1);
   // 选中当前物品
   int max_choose = knap2(remain_sum-weight[good_index], weight, profit, good_index-1)+profit[good_index];
   if(max_choose>max_no_choose){
     printf("%d ", weight[good_index]);
     return max_choose;
   }else{
     return max_no_choose;
   }
   //return max(max_choose, max_no_choose);
}

/*
0/1背包是NP-hard问题,不存在多项式解
给定一个背包容量为S, 有n件物品,重量分别为w1,w2,...,wn, 利润分别为:p1,p2,...,pn,
从中选取若干物品使得总利润最大,返回最大利润以及其选品方案


可以使用回溯法,不同于递归算法,递归算法是找大规模问题与小规模问题的关系,回溯法是对问题解空间进行搜索的算法
n件物品的搜索空间是:2^n,用1表示取,0表示不取

注意:搜索过程中要累加所取物品的重量,回溯法还要做现场清理,也就是将当前物品不取,并且从累加重量中减去当前物品的重量
*/

void knap3(int remain_weight, int current_profit, int& max_profit, int*weight, int* profit, int* used, int* max_used, int current_index, int length){
   if(current_index==length){ // 搜索到达叶结点
        if(current_profit>max_profit) {
            max_profit=current_profit;
            // 将最佳方案保存
            for(int i=0;i<length;i++) {
                max_used[i]= used[i];
            }
        }
        return;
   }// if

   if(weight[current_index]<=remain_weight){
       used[current_index] = 1; // 装入第i件物品
       // 测试下一个商品
       knap3(remain_weight - weight[current_index], current_profit+profit[current_index], max_profit, weight, profit, used, max_used, current_index+1, length);
       // 回溯之前清理现场
       used[current_index] = 0; // 不装入第i件物品
   }
   // 不装入第i件物品
   used[current_index] = 0 ;
   knap3(remain_weight, current_profit, max_profit, weight, profit, used, max_used, current_index+1, length);
}


/*
0/1背包是NP-hard问题,不存在多项式解
给定一个背包容量为S, 有n件物品,重量分别为w1,w2,...,wn, 利润分别为:p1,p2,...,pn,
从中选取若干物品使得总利润最大,返回最大利润以及其选品方案

此法将加入启发式搜索剪枝,利用bound进行剪枝,需要事先将物品按 利润/重量 的性价比 降序排序后再用, 时间复杂度仍为O(n)

*/
struct Goods{
    int weight;
    int profit;
};

// 计算不装物品i的利润上限
float profit_bound(int current_profit,
                   int current_index,
                   int remain_weight,
                   Goods*goods,
                   int length){

    float sum_profit=current_profit;
    // 从下一物品开始算
    for(int i=current_index+1;i<length;i++){
        if(remain_weight>goods[i].weight) {
           sum_profit += goods[i].profit;
           remain_weight-=goods[i].weight;
        }else{ // remain_weight< goods.weight
          float last_profit = remain_weight*1.0/goods[i].weight*goods[i].profit; // 最后剩余容量的利润
          return sum_profit + last_profit;
        }
    }
}

void knap4(int remain_weight, int current_profit, int& max_profit, Goods* goods, int* used, int* max_used, int current_index, int length){
   if(current_index==length){ // 搜索到达叶结点
        if(current_profit>max_profit) {
            max_profit=current_profit;
            // 将最佳方案保存
            for(int i=0;i<length;i++) {
                max_used[i]= used[i];
            }
        }
        return;
   }// if

   if(goods[current_index].weight<=remain_weight){
       used[current_index] = 1; // 装入第i件物品
       // 测试下一个商品
       knap4(remain_weight - goods[current_index].weight, current_profit+goods[current_index].profit, max_profit, goods, used, max_used, current_index+1, length);
       // 回溯之前清理现场
       used[current_index] = 0; // 不装入第i件物品
   }
   // 计算不选i的收益上界
   float upper_profit_bound = profit_bound(current_profit, current_index, remain_weight, goods, length);
   // 不选i的收益上界 > max_profit时,才执行不选i
   if(upper_profit_bound>max_profit){
       // 不装入第i件物品
       used[current_index] = 0 ;
       knap4(remain_weight, current_profit, max_profit, goods, used, max_used, current_index+1, length);
   }
}

// 利用循环代替递归(目前有些问题)
void knap5(int max_weight,int& max_profit, Goods* goods, int*max_used, int length) {
    int i=0;
    int sum_weight=0;
    int sum_profit=0;
    int* used = new int[length];
    for(int j=0;j<length;j++){
       used[j]=0;
    }
    while(true){
        // 选取物品
        while(i<length&&sum_weight+goods[i].weight<=max_weight){
             used[i] = 1;
             sum_weight+=goods[i].weight;
             sum_profit+=goods[i].profit;
             i++;
        }
       // 搜索到叶结点且利润更高,更新max_profit
       if(i==length&&sum_profit>max_profit){
            max_profit=sum_profit;
            for(int j=0;j<length;j++){
               max_used[j] = used[j];
            }
       } else { // 到了叶结点且利润低 或者 重量已超
           used[i] = 0; // 容量超出
       }
       // 如果不选i的利润 < max_profit
       while((max_weight-sum_weight>0)&&profit_bound(sum_profit, i, max_weight-sum_weight, goods, length)<=max_profit){
          // 找最后选取的物品
          while(i>=0&&used[i]==0){
             i--;
          }
          if(i==-1){ // 回溯到树根,结束搜索
            return;
          }
          // 否则回溯,搜索不选第i件物品的情况
          used[i] = 0;
          sum_weight-=goods[i].weight;
          sum_profit-=goods[i].profit;
       };
       i++;
    }// while

    delete[] used;
}

int main(int argc, char* argv[])
{

    {
        printf("knap1");
        int good_weight[] = {1,2,3,4,5,6}; // 每个商品的重量
        int good_sum = 12; // 总重量
        int len = sizeof(good_weight)/sizeof(int);
        bool has_sum = knap1(good_sum, good_weight, len-1);// 这里找到一个就停止了,并不会找到所有的背包组合
        if(!has_sum){
            printf("没有合法的背包!");
        }
    }

    {
        printf("\nknap2\n");
        int good_weight[] = {1,2,3,4,5,6}; // 每个商品的重量
        int good_profit[] = {3,5,3,4,3,2}; // 每个商品的利润
        int good_sum = 12; // 总重量
        int len = sizeof(good_weight)/sizeof(int);
        int max_profit = knap2(good_sum, good_weight, good_profit, len-1);// 这里找到一个就停止了,并不会找到所有的背包组合
        printf("最大利润:%d len:%d", max_profit, len);
    }

    {
        printf("\nknap3\n");
        int good_weight[] = {1,2,3,4,5,6}; // 每个商品的重量
        int good_profit[] = {3,5,3,4,3,2}; // 每个商品的利润
        int good_sum = 12; // 总重量
        int len = sizeof(good_weight)/sizeof(int);
        int* used=new int[len];
        int* max_used = new int[len];
        for(int i=0;i<len;i++){
            used[i]=0;
            max_used[i]=0;
        }
        int max_profit =0;
        knap3(good_sum, max_profit, max_profit, good_weight, good_profit,used, max_used, 0, len);// 这里找到一个就停止了,并不会找到所有的背包组合
        printf("最大利润:%d 最大利润方案:", max_profit);
        for(int i=0;i<len;i++){
            if(max_used[i]){
                printf("%d ", good_weight[i]);
            }
        }
        printf("\n选中商品利润:");
        for(int i=0;i<len;i++){
            if(max_used[i]){
                printf("%d ", good_profit[i]);
            }
        }

        delete[] used;
        delete[] max_used;
    }

   {
        printf("\n\nknap4");
        Goods goods[] = { {1,3},
                          {6,2},
                          {3,3},
                          {2,5},
                          {4,4},
                          {5,3},
                          }; // 每个商品的重量
        int good_sum = 12; // 总重量

        int len = sizeof(goods)/sizeof(Goods);
        int* used=new int[len];
        int* max_used = new int[len];
        for(int i=0;i<len;i++){
            used[i]=0;
            max_used[i]=0;
        }
        int max_profit =0;
        printf("\n排序前weights:");
        for(int i=0;i<len;i++){
            printf("%d ", goods[i].weight);
        }
        sort(goods,goods+len,[=](Goods a, Goods b){ return a.profit*1.0/a.weight>b.profit*1.0/b.weight; }); // 调用c++默认的排序函数进行排序
        printf("\n性价比降序:");
        for(int i=0;i<len;i++){
            printf("%.4f ", goods[i].profit*1.0/goods[i].weight);
        }
        printf("\nweights:");
        for(int i=0;i<len;i++){
            printf("%d ", goods[i].weight);
        }
        printf("\nprofits:");
        for(int i=0;i<len;i++){
            printf("%d ", goods[i].profit);
        }

        knap4(good_sum, max_profit, max_profit, goods, used, max_used, 0, len);// 这里找到一个就停止了,并不会找到所有的背包组合

        printf("\n最大利润:%d 最大利润方案:", max_profit);
        for(int i=0;i<len;i++){
            if(max_used[i]){
                printf("%d ", goods[i].weight);
            }
        }
        printf("\n选中商品利润:");
        for(int i=0;i<len;i++){
            if(max_used[i]){
                printf("%d ", goods[i].profit);
            }
        }

        delete[] used;
        delete[] max_used;
    }

   {
        printf("\n\nknap5");
        Goods goods[] = { {1,3},
                          {6,2},
                          {3,3},
                          {2,5},
                          {4,4},
                          {5,3},
                          }; // 每个商品的重量
        int good_sum = 12; // 总重量

        int len = sizeof(goods)/sizeof(Goods);
        int* max_used = new int[len];
        for(int i=0;i<len;i++){
            max_used[i]=0;
        }
        int max_profit =0;
        printf("\n排序前weights:");
        for(int i=0;i<len;i++){
            printf("%d ", goods[i].weight);
        }
        sort(goods,goods+len,[=](Goods a, Goods b){ return a.profit*1.0/a.weight>b.profit*1.0/b.weight; }); // 调用c++默认的排序函数进行排序
        printf("\n性价比降序:");
        for(int i=0;i<len;i++){
            printf("%.4f ", goods[i].profit*1.0/goods[i].weight);
        }
        printf("\nweights:");
        for(int i=0;i<len;i++){
            printf("%d ", goods[i].weight);
        }
        printf("\nprofits:");
        for(int i=0;i<len;i++){
            printf("%d ", goods[i].profit);
        }

        knap5(good_sum, max_profit, goods, max_used, len);// 这里找到一个就停止了,并不会找到所有的背包组合

        printf("\n最大利润:%d 最大利润方案:", max_profit);
        for(int i=0;i<len;i++){
            if(max_used[i]){
                printf("%d ", goods[i].weight);
            }
        }
        printf("\n选中商品利润:");
        for(int i=0;i<len;i++){
            if(max_used[i]){
                printf("%d ", goods[i].profit);
            }
        }

        delete[] max_used;
    }
    return 0;
}
