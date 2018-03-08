#include <algorithm>
#include <iterator>
#include <iostream>
using namespace std;

// 调整大顶堆（仅是调整过程，建立在大顶堆已构建的基础上）
void adjustHeap(int arr[],int i,int length){
    int temp = arr[i];//先取出当前元素i
    // 向下查找它的位置
    for(int child=i*2+1;child<length;child=child*2+1){//从i结点的左子结点开始，也就是2i+1处开始
        if(child+1<length && arr[child]<arr[child+1]) child++;  //如果左子结点小于右子结点，child指向右子结点
        if(arr[child] > temp){//如果子节点大于父节点，将子节点值赋给父节点（不用进行交换）
            arr[i] = arr[child];
            i = child;
        }else{ // 若子结点小于父节点,找到当前元素的位置，停止
            break;
        }
    }
    arr[i] = temp;//将temp值放到最终的位置
}

void heapSort(int arr[], int length){
    //1.构建大顶堆
    for(int i=length/2-1;i>=0;i--){
        //从第一个非叶子结点从下至上，从右至左调整结构
        adjustHeap(arr,i,length);
    }
    //2.调整堆结构+交换堆顶元素与末尾元素
    for(int j=length-1;j>0;j--){
        swap(arr[0],arr[j]);//将堆顶元素与末尾元素进行交换，排好序的元素都将放在链表末尾
        adjustHeap(arr,0,j);//重新对堆进行调整
    }
}

int main()
{
    int a[] = {23,89,127,-6,-9, 7, 8, 13, 24 , 11, 19, 9, 54, 6, 4, 11, 1, 2, 33}; // 1, 2, 4, 6, 7, 8, 9, 11, ...
    int len = sizeof(a) / sizeof(int);
    cout<<"before sorted:"<<endl;
    copy(a, a+len, std::ostream_iterator<int>(std::cout, " ")); // algorithm,iterator
    cout<<"\n";
    heapSort(a,len);
    std::cout<<"after sorted:"<<std::endl;
    copy(a, a+len, std::ostream_iterator<int>(std::cout, " ")); // algorithm,iterator
    cout<<"\n";
    return 0;
}
