/**
今日头条算法面试题 20180511

一个服务在调用数据库的时候需要设置一个合理的最大连接数，现在假设已经统计了一段时间里服务的连接数据库状态信息，
得到一个三元组列表[{start_time, end_time, link_id}]，三元组中每个字段分别表示一个连接开始建立的时间，
结束的时间，连接的标识，再假设合理的最大连接数就是服务在这段时间内同时连接数据库的总连接数加1，
那么我们应该设置成多大的连接数比较合理？

假设这些三元组已经按时间顺序有序.

 s1-------------e1
        s2--------------e2
    s3---------e3
          s4--------------e4
                s5------------------e5
按开始时间排序后,并不会改变连接数：
 s1-------------e1
    s3---------e3
        s2--------------e2
          s4--------------e4
                s5------------------e5

思路1:
1.先对所有记录按开始时间排序
2.当遍历到某条记录时, cur_cnt++，将记录的end_time加入一个有序队列（插入排序）
3.判断当前开始时间是否大于end_time里最小的元素，若是将所有end_time小于cur_time的元素出队，同时 cur_cnt --
4.max_cnt = std::(max_cnt, cur_cnt)

思路2(比第一种方法好理解):
1.先对所有记录按开始时间排序,
2.每遍历一个数, 则检查开始或者结束状态,如果是开始则+1,如果是结束则-1,并记录当前值cur_cnt
3.max_cnt =std::(max_cnt, cur_cnt)


*/

#include <iostream>
#include <algorithm>
#include <cmath>
using namespace std;

struct Link{
    int start_time;
    int end_time;
    int id;
};

// 排序队列, qs指向第一个元素，qe指向最后一个元素的下一个
const int N =100;
int queue[N];
int qs=0, qe=0; // 头尾指针相等，说明队列为空

void print_arr(Link arr[], int len){
    for(int i=0;i<len;i++){
        Link ele = arr[i];
        cout<<"id:"<<ele.id<< " start_time:"<<ele.start_time<< " end_time:"<< ele.end_time <<endl;
    }
}

// 将结束时间插入队列，返回插入位置
int insert_queue(int x){
    int i;
    if(qe==qs){
       i = qs;
    }else{
       i=qe-1;
       // 寻找插入位置
       while(i>0&&queue[i]>x) {
            queue[i+1] = queue[i];
            i--;
       }
    }
   queue[i]=x;
   qe++;
   return i;
}

// 出队所有小于x的元素,返回出队的元素个数
int dequeue(int x){
    int old = qs;
    while(queue[qs]<x) {
        qs++;
    }
   return qs - old;
}


// 1.先对所有记录按开始时间排序
// 2.当遍历到某条记录时, cur_cnt++，将记录的end_time加入一个有序队列（插入排序）
// 3.判断当前开始时间是否大于end_time里最小的元素，若是将所有end_time小于cur_time的元素出队，同时 cur_cnt --
// 4.max_cnt =std::(max_cnt, cur_cnt)
int max_conn(Link arr[], int len){
    int max_cnt = 0;
    std::sort(arr, arr + len, [](Link a, Link b){ return a.start_time<=b.start_time; }); // 调用c++默认的排序函数进行排序
    cout<<"arr after sorted"<<endl;
    print_arr(arr,len);
    //
    for(int i=0;i<len;i++){
       insert_queue(arr[i].end_time);
       dequeue(arr[i].start_time);
       max_cnt = std::max(max_cnt, qe-qs);
    }
    return max_cnt;
}


int main(){
    Link arr[]={
        {1,10,1},
        {5,15,2},
        {3,8,3},
        {6,18,4},
        {12,20,5},
    };
    int len = sizeof(arr)/sizeof(Link);
    print_arr(arr,len);
    std::cout<<"max_count:"<< max_conn(arr, len) <<std::endl;
    return 0;
}
