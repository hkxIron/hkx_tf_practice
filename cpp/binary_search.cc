/*
  g++ -std=c++11 -o  binary_search binary_search.cc &&./binary_search
*/
// 插入排序
// 被插入表的已经是有序，待插入元素每次需要查找插入位置，同时需要移动元素
#include<iostream>
#include<iterator>
#include<algorithm>
using namespace std;

int binary_search(int a[],int n,int key){
    int low = 0;
    int high = n-1;
    while(low<=high){ // 注意，这里需要相等
        int mid = (low+high)/2;
        if(key<a[mid]) high=mid-1; // 因为值已经不相等了，所以要-1
        else if(key>a[mid]) low = mid +1;
        else return mid;
    }
    // 最后如果未找到,就是 low = high +1
    return -1; // 未找到，返回-1
}

/**
头条面试:
给一个非降序数组,要求找出数组中最后一个小于x的数, 如果没有就返回-1

index 0  1 2 3 4  5
data  -1 0 5 7 10 15

*/
int binary_search_last_less_x(int a[], int n, int key) {
    int low = 0;
    int high = n-1;
    while (low<=high) { // 注意，这里需要相等
        int mid = (low+high)/2;
        if(key<=a[mid]) high=mid-1; // 因为值已经不相等了，所以要-1
        else low = mid +1;
        //else if (key>a[mid]) low = mid +1;
    }
    // 最后如果未找到,就是 low = high +1
    /**
    if(low==0) return -1;
    else if(high ==n-1) return n-1;
    else {
        return low-1;
    }
    */
    return low-1;
    //return -1; // 未找到，返回-1
}

int binary_search_last_less_x2(int a[], int n, int x) {
    int left = 0;
    int right = n-1;
    if (a[0]>=x||a[right]>=x) {
        return -1;
    }
    // 现在里面肯定存在一个小于x的数
    while (left<=right) {
       int mid = (left+(right - left))/2;
       if(a[mid]<x){
           left = mid+1;
       }else if(a[mid]==x){ // a[mid]==x 依然不满足<x, 因此说明小于是在左边界, 往左移动
           right = mid-1;
       }else if(a[mid]>x){ // a[mid]>x,依然不满足<x, 继续往左查找
           right = mid-1;
       }
    }
    return right;  // left = right+1;
}

int main(){
    {
        //       0  1   2 3 4 5
        int a[]={-3,-2,-1,4,5,9};
        int n = sizeof(a)/sizeof(int);
        std::copy(a, a+n, std::ostream_iterator<int>(std::cout, " ")); // algorithm,iterator
        //
        std::cout<<endl;
        std::cout<<binary_search(a,n,45)<<endl;
        std::cout<<binary_search(a,n,10)<<endl;
        std::cout<<binary_search(a,n,4)<<endl;
        std::cout<<binary_search(a,n,9)<<endl;
        std::cout<<binary_search(a,n,-3)<<endl;
    }
    {
        cout << " 找最后一个小于x的数: " <<endl;
        int a[] = {1,2,3,5,6};
        int target[] = {-1, 1, 3, 10};
        int n = sizeof(a)/sizeof(int);
        int target_num = sizeof(target)/sizeof(int);
        for(int i=0;i<n;i++){
            cout<<a[i]<<" ";
        }
        cout<<std::endl;
        for(int i=0;i<target_num;i++){
            cout<<"x:"<< target[i]<< " find num:" <<binary_search_last_less_x(a, n, target[i]) << endl;
        }
    }

    {
        cout << " 找最后一个小于x的数: " <<endl;
        int a[] = {1,2,3,5,6};
        int target[] = {-1, 1, 3, 10};
        int n = sizeof(a)/sizeof(int);
        int target_num = sizeof(target)/sizeof(int);
        for(int i=0;i<n;i++){
            cout<<a[i]<<" ";
        }
        cout<<std::endl;
        for(int i=0;i<target_num;i++){
            cout<<"x:"<< target[i]<< " find num:" <<binary_search_last_less_x(a, n, target[i]) << endl;
        }
    }
}