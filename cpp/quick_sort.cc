#include <iostream>
#include <stdlib.h>
using namespace std;

/**
*  交换顺序表a中的区间[low,hight]中的记录，枢轴记录到位，并返回其所在的位置，
*  此时，在它之前的记录均小于它，在它之后的均大于它
*/
int partition(int a[], int low, int high){
    int pivot = a[low] ;
    while(low<high){
        while(low<high&&a[high]>=pivot) --high; // 将右边小于pivot的找出来
        a[low] = a[high];
        while(low<high&&a[low]<=pivot) ++low; // 将左边大于pivot的找出来
        a[high] = a[low];
    }
    a[low] = pivot;
    return low; // 此时low与high相等
}

void quickSort(int a[], int low,int high){
    if(low<high){
       int pivotIndex = partition(a,low,high);// 一次划分后，左边的都比它小，右边的都比它大
       quickSort(a,low,pivotIndex-1);
       quickSort(a,pivotIndex+1,high);
    }
}


int main()
{
    int a[] = {7, 8, 13, 24 , 11, 19, 9, 54, 6, 4, 11, 1, 2, 33}; // 1, 2, 4, 6, 7, 8, 9, 11, ...
    int len = sizeof(a) / sizeof(int);
    cout<<"before sorted:"<<endl;
    for(int i = 0; i< len; i ++)
        std::cout<< a[i]<<" "; // 第4小的数在a[3] = 6
    cout<<"\n";
    quickSort(a,0,len-1);
    std::cout<<"after sorted:"<<std::endl;
    for(int i = 0; i< len; i ++)
        std::cout<< a[i]<<" "; // 第4小的数在a[3] = 6
    cout<<"\n";
    return 0;
}
