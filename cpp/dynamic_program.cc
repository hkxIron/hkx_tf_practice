/*
g++ -std=c++11 -o bin dynamic_program.cc &&./bin
*/

#include<cstdio>
#include<stdlib.h>
#include<iostream>
#include<cstring>
#include<algorithm>
#include<vector>
#include<assert.h>
using namespace std;

void print_vec(vector<int>& v, bool append_new_line=true) {
    int N = v.size();
    for(int i=0;i<N;i++) {
        std::cout<<v[i]<<" ";
    }
    if(append_new_line) std::cout<<std::endl;
}

void print_vec(vector<vector<int>>& v) {
    for(vector<int> arr:v) print_vec(arr);
}

/**
leetcode 416:
 给定一个只包含正整数的非空数组, 是否可以将这个数据划分成两个子集,使得两个子集的元素之各相等
 1.每个数组的元素<100
 2.数组的大小不会超过100
*/
// 输⼊⼀个集合， 返回是否能够分割成和相等的两个⼦集
bool canPartition(vector<int>& nums) {
    int sum = 0;
    for(int num: nums) sum+=num;
    // 和为奇数时, 不可能划分为两个和相等的集合
    if (sum&1) return false;
    int N = nums.size();
    sum = sum>>1; // 一半
    // dp[i][j]: 选前i个元素中的某些之和是否为j
    vector<vector<bool>> dp(N+1, vector<bool>(sum+1, false));
    // base case
    for(int i = 0; i <= N; i++) {
        dp[i][0] = true; // 前i个元素都不选,可以凑成0
    }
    for (int i = 1; i <= N; i++) {
        int curr_num = nums[i-1];
        for (int j = 1; j <= sum; j++) { // 容量为j时
            if (j < curr_num) {
                // 背包容量不⾜， 不能装⼊第 i 个物品
                dp[i][j] = dp[i - 1][j];
            } else {
                // 1. 不装入
                // 2. 装入curr_num
                dp[i][j] = dp[i - 1][j] \
                         | dp[i - 1][j-curr_num];
            }
        }
    }
    return dp[N][sum];
}

// 进行状态压缩
bool canPartition2(vector<int>& nums) {
    int sum = 0;
    for(int num: nums) sum+=num; // 可以这样遍历数组
    // 和为奇数时, 不可能划分为两个和相等的集合
    if (sum&1) return false;
    int N = nums.size();
    sum = sum>>1; // 一半
    // dp[i][j]: 选前i个元素中的某些之和是否为j
    vector<bool> dp(sum+1, false);
    // base case
    dp[0] = true;
    for (int i = 0; i < N; i++)
        // 唯⼀需要注意的是 j 应该从后往前反向遍历， 因为每个物品（或者说数字） 只能⽤⼀次， 以免之前的结果影响其他的结果。
        for (int j = sum; j >= 0; j--)
            if (j - nums[i] >= 0)
                // 1. 不装入
                // 2. 装入curr_num
                dp[j] = dp[j] || dp[j - nums[i]];
    return dp[sum];
}

int min(int a, int b, int c) {
//    int ab = a>=b?a:b;
//    int bc = b>=c?b:c;
//    return ab?ab:bc;
    return std::min(a, std::min(b,c));
}
// 两个串的最小编辑距离
int minEditDistance(string s1, string s2) {
    int m = s1.length(), n = s2.length();
    //int[][] dp = new int[m + 1][n + 1];
    vector<vector<int>> dp(m+1, vector<int>(n+1, 0));
    // base case
    for (int i = 1; i <= m; i++) dp[i][0] = i;
    for (int j = 1; j <= n; j++) dp[0][j] = j;
    // ⾃底向上求解
    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (s1[i-1] == s2[j-1])
                dp[i][j] = dp[i - 1][j - 1];
            else
                dp[i][j] = min(dp[i - 1][j] + 1,  // 插入
                               dp[i][j - 1] + 1, // 删除
                               dp[i-1][j-1] + 1); // 替换
        }
    }

    std::cout<<"dp table:"<<std::endl;
    print_vec(dp);
    // 储存着整个 s1 和 s2 的最⼩编辑距离
    return dp[m][n];
}


void test_canPartition() {
    vector<int> arr = {1,2,3,4, 10, 3, 7}; // 每个商品的重量
    bool flag = canPartition(arr);
    print_vec(arr);
    std::cout<<"flag:"<<flag<<std::endl;
}

void test_canPartition2() {
    vector<int> arr = {1,2,3,4, 10, 3, 7}; // 每个商品的重量
    bool flag = canPartition2(arr);
    print_vec(arr);
    std::cout<<"flag:"<<flag<<std::endl;
}

void test_minEditDistance() {
    {
        std::string s1 = "intention";
        std::string s2= "execution";
        std::cout<<"s1:"<<s1<<std::endl;
        std::cout<<"s2:"<<s2<<std::endl;
        std::cout<<"minEditDistance:"<<minEditDistance(s1, s2)<<std::endl;
    }
    {
        std::string s1 = "horse";
        std::string s2= "ros";
        std::cout<<"s1:"<<s1<<std::endl;
        std::cout<<"s2:"<<s2<<std::endl;
        std::cout<<"minEditDistance:"<<minEditDistance(s1, s2)<<std::endl;
    }
}

int main(int argc, char* argv[])
{
    test_canPartition();
    test_canPartition2();
    test_minEditDistance();
    return 0;
}
