/*
g++ -std=c++11 -o knap knap.cc &&./knap
*/

#include <cstdio>
#include <stdlib.h>
#include <iostream>
#include <cstring>
#include<algorithm>
#include<vector>
#include <assert.h>
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

   // 当前容量可以装下物品,尝试装入
   if(weight[current_index]<=remain_weight){
       used[current_index] = 1; // 装入第i件物品
       // 测试下一个商品
       knap3(remain_weight - weight[current_index], current_profit+profit[current_index], max_profit, weight, profit, used, max_used, current_index+1, length);
       // 回溯之前清理现场
       used[current_index] = 0; // 不装入第i件物品
   }
   // 即使当前容量可以装下物品,也可以选择不装入, 因此不能是else
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

// 用动态规划解决knap问题, 将物品填入背包,使背包所装物品的价值最大,如果没有则返回0
// wt: 每个物品的重量
// val: 每个物品的价值
// W: 背包最大重量
// N: 物品个数
int knapsack_dp(int max_weight, vector<int>& weights, vector<int>& values) {
    // base case 已初始化
    int N = weights.size();
    // dp[i][w] 表⽰：对于前 i 个物品，当前背包的容量为w时，这种情况下可以装下的最⼤价值是 dp[i][w]
    // 其每一行代表选择前i个物品,每一列代表当前的背包容量w
    // ⽐如说， 如果 dp[3][5] = 6 ， 其含义为：对于给定的⼀系列物品中，
    // 若只对前3个物品进⾏选择，当背包容量为5时，最多可以装下的价值为 6
    vector<vector<int>> dp(N + 1, vector<int>(max_weight + 1, 0));
    // 准确的说是经过了前i个物品,即这些物品有可能被选中, 也有可能未被选中
    for (int i = 1; i <= N; i++) {
        for (int w = 1; w <= max_weight; w++) {
            // w为当前的背包容量, weights[i-1],为第i个物品的重量
            if (w < weights[i-1]) {
                // 背包容量<物品重量, 不装入w[i-1], 价值不变
                dp[i][w] = dp[i - 1][w];
            } else {
                // 装⼊或者不装⼊背包， 择优
                dp[i][w] = std::max(dp[i - 1][w - weights[i-1]] + values[i-1],  // 装入第i个物品,加上前i-1个物品装下时的价值
                                    dp[i - 1][w]); // 不装入
            }
        }
    }
    // 背包容量
    std::cout<<"weights:\n  ";
    for(int i=0;i<N;i++){
       std::cout<<weights[i]<<" ";
    }
    std::cout<<"\nvalues:\n  ";
    for(int i=0;i<N;i++){
       std::cout<<values[i]<<" ";
    }

    std::cout<<"\ndp_table:\n";
    // 输出dp table
    for(int i=0;i<N+1;i++){
        for(int j=0;j<max_weight+1;j++) {
            std::cout<<dp[i][j] <<" ";
        }
        std::cout<<std::endl;
    }
    return dp[N][max_weight];
}

/**
  凑零钱问题
  给定不同面额硬币以及总金额,求可以凑成总金额的组合数(假设每种面额的硬币有无限个)
*/
int coin_change(int amount, vector<int> coins) {
    int N = coins.size(); // 1,2,5
    // 若只使⽤前i个物品，当背包容量为j时， 有 dp[i][j] 种⽅法可以装满背包
    vector<vector<int>> dp(N + 1, vector<int>(amount + 1, 0));
    // base case为dp[0][..] = 0,dp[..][0] = 1 。
    // 因为如果不使⽤任何硬币⾯值，就⽆法凑出任何⾦额
    // 如果凑出的⽬标⾦额为0，那么“⽆为⽽治”就是 唯⼀的⼀种凑法
    for(int i=0;i<=N;i++) {
        dp[i][0] = 1;
    }
    // 若只使⽤前i个物品，当背包容量为j时， 有 dp[i][j] 种⽅法可以装满背包
    for (int i = 1; i <= N; i++) {
        int curr_coin = coins[i-1]; // 当前的硬币大小
        for (int j = 1; j <= amount; j++) { // 当前要凑的钱数j
            // dp[i][j-coins[i-1]] 也不难理解，如果你决定使⽤这个⾯值的硬币，那么就应该关注如何凑出⾦额 j - coins[i-1]
            if (j >= curr_coin) { // 要凑的钱数>=当前币数,因此可以用当前coin来凑
                dp[i][j] = dp[i - 1][j] // 不使用第i个coin, 使⽤前i-1个coin可以凑出j
                         + dp[i][j - curr_coin]; // 使用第i个coin, 凑出余下的j-coins[i-1]
            } else { // 要凑的钱数<当前币数,无法使用当前coin来凑, 使用前i-1个币来凑
                dp[i][j] = dp[i - 1][j];
            }
        }
    }
    // 背包容量
    std::cout<<"coins:\n  ";
    for(int i=0;i<N;i++){
       std::cout<<coins[i]<<" ";
    }

    std::cout<<"\ndp_table:\n";
    // 输出dp table
    for(int i=0;i<N;i++){
        for(int j=0;j<=amount;j++) {
            std::cout<<dp[i][j] <<" ";
        }
        std::cout<<std::endl;
    }
    return dp[N][amount];
}

int coin_change2(int amount, vector<int> coins) {
    int N = coins.size();
    // 若只使⽤前i个物品，当背包容量为j时， 有 dp[i][j] 种⽅法可以装满背包
    vector<int> dp(amount + 1, 0); // dp[j]:凑出面额为j的方法个数
    // base case: 凑出面额为0的方法只有1种,即都不用是一种方法
    // 若只使⽤前i个物品，当背包容量为j时， 有 dp[i]种⽅法可以装满背包
    dp[0] = 1;
    for (int i = 0; i < N; i++) {
        for (int j = 1; j <= amount; j++) { // 当前要凑的钱数j
            if (j >= coins[i]) { // 要凑的钱数j >=当前币数,因此可以用当前coin来凑
                dp[j] = dp[j]  // 不使用coin[i], 凑出余下的j-coins[i-1]
                        + dp[j-coins[i]]; //使用coin[i]
            }
        }
    }
    std::cout<<"coins:\n  ";
    for(int i=0;i<N;i++){
       std::cout<<coins[i]<<" ";
    }
    std::cout<<"\ndp table:\n";
    for(int i=0;i<=amount;i++){
       std::cout<<"x:"<<i<<" n:"<<dp[i]<<" ";
    }
    std::cout<<std::endl;
    return dp[amount];
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
        printf("\nknap_dp(动态规划解背包问题)\n");
        vector<int> good_weight = {1,2,3,4,5,6}; // 每个商品的重量
        vector<int> good_profit = {3,5,3,4,3,2}; // 每个商品的利润
        int good_sum = 12; // 背包总重量
        int len = good_weight.size();
        assert(good_weight.size() == good_profit.size());
        int max_profit = knapsack_dp(good_sum, good_weight, good_profit);// 这里找到一个就停止了,并不会找到所有的背包组合
        printf("最大利润:%d len:%d", max_profit, len);
    }
    {
        printf("\nknap_dp(动态规划解凑零钱问题)\n");
        vector<int> coins = {1, 2, 5}; // 每个商品的重量
        int amount = 5;
        // 5=1+1+1+1+1
        // 5=1+1+1+2
        // 5=1+2+2
        // 5=5
        int num = coin_change(amount, coins); // 感觉有点问题吧
        int num2 = coin_change2(amount, coins); // 感觉有点问题吧
        printf("凑钱的方法数:%d 凑法2:%d", num, num2);
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
                          }; // 每个商品的重量,价值
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
