#include <algorithm>
#include <iterator>
#include <iostream>
using namespace std;

// 大顶堆最后排出的是升序序列
// blog:https://www.cnblogs.com/chengxiao/p/6129630.html
/*
堆排序

　　堆排序是利用堆这种数据结构而设计的一种排序算法，堆排序是一种选择排序，它的最坏，最好，平均时间复杂度均为O(nlogn)，它也是不稳定排序。首先简单了解下堆结构。
堆

　　堆是具有以下性质的完全二叉树：每个结点的值都大于或等于其左右孩子结点的值，称为大顶堆；
或者每个结点的值都小于或等于其左右孩子结点的值，称为小顶堆。
同时，我们对堆中的结点按层进行编号，将这种逻辑结构映射到数组中就是下面这个样子
                50 (0)
               /   \
              45    40 (1,2)
             / \    / \
            20  25 35  30  (3,4,5,6)
           / \
          10  15  (7,8)

ind: 0  1  2  3  4  5  6  7  8
arr: 50 45 40 20 25 35 30 10 15

该数组从逻辑上讲就是一个堆结构，我们用简单的公式来描述一下堆的定义就是：
大顶堆：arr[i] >= arr[2i+1] && arr[i] >= arr[2i+2]
小顶堆：arr[i] <= arr[2i+1] && arr[i] <= arr[2i+2]

堆排序是一种选择排序，整体主要由构建初始堆+交换堆顶元素和末尾元素并重建堆两部分组成。其中构建初始堆经推导复杂度为O(n)，在交换并重建堆的过程中，需交换n-1次，而重建堆的过程中，
根据完全二叉树的性质，[log2(n-1),log2(n-2)...1]逐步递减，近似为nlogn。所以堆排序时间复杂度一般认为就是O(nlogn)级。
*/

// 调整大顶堆（仅是调整过程，建立在大顶堆已构建的基础上）
void adjustHeap(int arr[],int i,int length){
    int temp = arr[i];//先取出当前元素i
    // 向下查找它的位置
    for(int child=i*2+1;child<length;child=child*2+1){//从i结点的左子结点开始，也就是2i+1处开始，一直到 length
        if(child+1<length && arr[child]<arr[child+1]) child++;  //如果左子结点小于右子结点，child指向右子结点
        if(arr[child] > temp){ //如果子节点大于父节点，将子节点值赋给父节点（不用进行交换）
            arr[i] = arr[child];
            i = child;
        }else{ // 若子结点(两孩子中较大的结点)小于父节点,找到当前元素的位置，停止
            break;
        }
    }
    arr[i] = temp;//将temp值放到最终的位置
}

void heapSort(int arr[], int length){
    //1.构建大顶堆
    //从第一个非叶子结点len/2-1从下至上，从右至左调整结构
    for(int i=length/2-1;i>=0;i--){
        adjustHeap(arr,i,length); //i到i+len的范围内调整
    }
    //2.调整堆结构+交换堆顶元素与末尾元素(排序)
    for(int j=length-1;j>0;j--){
        std::swap(arr[0],arr[j]);//将堆顶元素a[0]与末尾元素a[j]进行交换，排好序的元素都将放在链表末尾
        adjustHeap(arr,0,j);//重新对堆进行调整（0~0+j）
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
