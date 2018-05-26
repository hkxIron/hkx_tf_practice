//copyright@ July
//July、updated，2011.05.25。
//blog: https://blog.csdn.net/v_july_v/article/details/6444021
/*
第一节、求子数组的最大和
3.求子数组的最大和
题目描述：
输入一个整形数组，数组里有正数也有负数。
数组中连续的一个或多个整数组成一个子数组，每个子数组都有一个和。
求所有子数组的和的最大值。要求时间复杂度为O(n)。

例如输入的数组为1, -2, 3, 10, -4, 7, 2, -5，和最大的子数组为3, 10, -4, 7, 2，
因此输出为该子数组的和18。

其实算法很简单，当前面的几个数，加起来后，b<0后，
把b重新赋值，置为下一个元素，b=a[i]。
当b>sum，则更新sum=b;
若b<sum，则sum保持原值，不更新。。July、10/31。

思想是：当累加a[i]时，必须保证pre_sum>=0，否则，重新开始计数

*/
#include <iostream>
using namespace std;
int beginIndex = -1;
int endIndex = -1;
int maxsum(int a[],int n)
{
    int max=a[0];       //全负情况，返回最大数
    int sum=0; // 前面元素的累加和
    for(int j=0;j<n;j++)
    {
        if(sum>=0)     //如果前面元素累加sum>=0的话，就加
            sum+=a[j];
        else{
            sum=a[j];  //如果sum<0了，断开连接，将a[j]置为开始
            beginIndex = j;
        }
        if(sum>max) {
            max=sum;  // 记录历史中出现最大的和
            endIndex=j;
        }
    }
    return max;
}

int main()
{
    {
        int a[]={-1,-2,-3,-4};
        int len = sizeof(a)/sizeof(int);
        cout<<"max sum:"<<maxsum(a,len)<<endl;
    }

    {
        int a[]={1, -2, 3, 10, -4, 7, 2, -5};
        int len = sizeof(a)/sizeof(int);
        cout<<"max sum:"<<maxsum(a,len)<<endl;
        cout<<"index:"<<beginIndex<<"~"<<endIndex<<endl;
    }
    return 0;
}