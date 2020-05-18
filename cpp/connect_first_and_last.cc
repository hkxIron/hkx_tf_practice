/*
  将链表的首尾依次相连
 输入: 1 2 3 4 5 6
 输出: 1 6 2 5 3 4

  g++ -std=c++11 -o bin connect_first_and_last.cc &&./bin
*/
#include<iostream>
using namespace std;

struct node{
  int x;
  node* next;
};

// 得到第k个结点, 第一个代表p
node* get_k_node(node*p, int k) {
    while(--k>0){
        p=p->next;
    }
    return p;
}

int get_len(node*p) {
   if(p==NULL) return 0;
   int len = 1;
   while(p->next){
       p=p->next;
       len++;
   }
   return len;
}

node* get_tail_node(node* p){
   if(p==NULL) return p;
   while(p->next){
       p=p->next;
   }
   return p;
}

node* connect_head_tail(node* pList) {
    if(pList==NULL||pList->next == NULL) return pList;
    // 求取list长度
    int len = get_len(pList);
    int last_left_cnt = len/2;
    if(len&0x1) { // 奇数
        last_left_cnt = (len+1)/2;
    }
    // 找到右边的第一个node, 以减小耗时
    node* first_right_node = get_k_node(pList, last_left_cnt+1);
    int right_node_cnt = len - last_left_cnt;
    // 重新连接
    node* p=pList;

    while(p->next!=first_right_node) {
       node* pNext = p->next;
       // 得到最右边的点
       node* right_node = get_k_node(first_right_node, right_node_cnt);
       p->next = right_node; //
       right_node->next=pNext;
       p = pNext;
       right_node_cnt--; // 右边元素减1
    }

    // 最后一个指针置空
    if(len&0x1){
        p->next = NULL;
    }else{
        p->next->next = NULL;
    }
    return pList;
}


void print(node* p){
    while(p){
        std::cout<< " " << p->x;
        p=p->next;
    }
    std::cout<<"\n";
}

// 将链表逆序: 1 -> 2-> 3-> 4-> 5
// 1  2,3,4,5
// 2,1  3,4,5
// 3,2,1  4,5
// 5 -> 4 -> 3 -> 2 -> 1
node* reverse_link_list(node *head) {
    node* reversedHead = NULL;
    node* p = head;
    node* prev = NULL;
    while(p!=NULL) {
        node* pNext = p->next;
        if(pNext==NULL) reversedHead = p;
        //改变指针
        p->next = prev;
        //移动
        prev=p;
        p=pNext;
    }
    return reversedHead;
}

// 使用反转法
// input: 1 -> 2 -> 3 | ->4 ->5 ->6
// left: 1 -> 2 ->3
// 右边反转: 6 -> 5 -> 4
//连接: 1 -> 6 -> 2 -> 5 -> 3 -> 4
node* connect_head_tail_by_reverse(node* pList) {
    if(pList==NULL||pList->next == NULL) return pList;
    // 求取list长度
    int len = get_len(pList);
    int last_left_cnt = len/2;
    if(len&0x1) { // 奇数
        last_left_cnt = (len+1)/2;
    }
    // 找到右边的第一个node, 以减小耗时
    node* last_left_node = get_k_node(pList, last_left_cnt);
    node* first_right_node = last_left_node->next;
    last_left_node->next = NULL; // 断开左边的最后一个结点的next指针

    // 反转右边的链表
    node* reversed_right_list = reverse_link_list(first_right_node);
    // 合并两个链表
    node* pL = pList;
    node* pR = reversed_right_list;
    while(pL&&pR) {
       node* pLNext = pL->next;
       node* pRNext = pR->next;

       pL->next = pR;
       pR->next = pLNext;

       pL = pLNext;
       pR = pRNext;
    }
    return pList;
}

node* get_list(int*a, int len) {
    node* head;
    node* pre = NULL;
    for(int i=0;i<len;i++){
        node* p = new node();
        p->x = a[i];
        if(pre==NULL) head = p;
        else pre->next = p;
        pre=p;
    }
    return head;
}

int main(){
    {
        int a[] = {1,2, 3, 4, 5 ,6 };
        int len = sizeof(a)/sizeof(int);
        node* head = get_list(a, len);
        print(head);
        print(connect_head_tail(head));
    }

    {
        std::cout<<"------------"<<std::endl;
        int a[] = {1,2, 3, 4, 5};
        int len = sizeof(a)/sizeof(int);
        node* head = get_list(a, len);
        print(head);
        print(connect_head_tail(head));
    }

    {
        std::cout<<"------------"<<std::endl;
        int a[] = {1,2, 3, 4, 5, 6};
        int len = sizeof(a)/sizeof(int);
        node* head = get_list(a, len);
        print(head);
        print(connect_head_tail_by_reverse(head));
    }
/**
 1 2 3 4 5 6
 1 6 2 5 3 4
------------
 1 2 3 4 5
 1 5 2 4 3
------------
 1 2 3 4 5 6
 1 6 2 5 3 4

    **/
}
