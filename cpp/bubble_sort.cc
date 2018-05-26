// 冒泡排序：先排好后面的元素
#include<iostream>
#include<iterator>
#include<algorithm>
using namespace std;

void bubble_sort(int a[],int n){
    for(int i=0;i<n-1;++i){ // 排序的趟数, N-1
        bool swap_flag = false;
        for(int j=0;j<n-i-1;++j){ // 先排好后面的元素
            if(a[j]>a[j+1]){ // 每次都将大的元素往后排
                std::swap(a[j],a[j+1]);
                swap_flag = true;
            }
        }
        if(!swap_flag) break; // 没有交换，说明所有的相邻元素均有序，排序已经完成，
    }
}

int main(){
    int a[]={-1, -1,3,-2,10,7,-4,-3};
    int len = sizeof(a)/sizeof(int);
    cout<<"before sort:"<<endl;
    copy(a, a+len, std::ostream_iterator<int>(std::cout, " ")); // algorithm,iterator
    bubble_sort(a,len);
    cout<<"\nafter sort:"<<endl;
    copy(a, a+len, std::ostream_iterator<int>(std::cout, " ")); // algorithm,iterator
}