/*Trie树(字典树) 2011.10.10*/
// g++ -std=c++11 -o trie.exe trie.cpp
/*
1.2 Tire树的用途

Tire树核心思想是空间换取时间，利用字符串的公共前缀来节省查询时间，常用于统计与排序大量字符串。其查询的时间复杂度是O（L），只与待查询串的长度相关。所以其有广泛的应用，下边简单介绍下Tire树的用途

 Tire用于统计：

 题目：给你100000个长度不超过10的单词。对于每一个单词，我们要判断他出没出现过，如果出现了，求第一次出现在第几个位置。

 解法 ：从第一个单词开始构造Tire树，Tire树包含两字段，字符与位置，对于非结尾字符，其位置标0，结尾字符，标注在100000个单词词表中的位置。
 对词表建造Tire树，对每个词检索到词尾，若词尾的数字字段>0,表示单词已经在之前出现过，否则建立结尾字符标志，下次出现即可得知其首次出现位置，
 便利词表即可依次计算出每个单词首次出现位置。复杂度为O（N×L）L为最长单词长度，N为词表大小

 Tire用于排序

 题目：对10000个英文名按字典顺序排序

 解法：建造Tire树，先序便利即可得到结果。
*/
#include<iostream>
#include<cstdlib> // NULL
#include"stdio.h"
#define MAX 26
using namespace std;

typedef struct TrieNode                     //Trie结点声明
{
    bool isWord;                            //标记该结点处是否构成单词,此处可以记录该单词第一次出现在文本中的位置
    struct TrieNode *next[MAX];            //儿子分支,注意，这里前面有个struct
}Trie;

void init_trie(Trie* root){
    if(root){
        for(int i=0;i<MAX;i++) {
            root->next[i]=NULL;
        }
        root->isWord=false;
    }
}

// 先序遍历(根左右)即是对字符串排序
void pre_order_visit(Trie* root,char* begin ,char* prefix){
   if(root==NULL) return;
    if(root->isWord){
       cout<<begin<<endl; // 这里会输出单词，也就是排序了
    }
    for(int i=0;i<MAX;i++)
    {
        *(prefix++) = char('a'+i);
        pre_order_visit(root->next[i],begin,prefix);
        *(--prefix)='\0';
    }
}

// s为单词"hello"
void insert(Trie *root,const char *s)     //将单词s插入到字典树中
{
    if(root==NULL||*s=='\0')
        return;
    int i;
    Trie *p=root;
    while(*s!='\0') // 不等于空字符
    {
        int offset = *s - 'a' ;
        if(p->next[offset]==NULL)        //如果不存在，则建立结点
        {
            Trie *temp=(Trie *)malloc(sizeof(Trie));
            init_trie(temp);
            p->next[offset]=temp;
        }
        p=p->next[offset];
        s++;
    }
    p->isWord=true;                       //单词结束的地方标记此处可以构成一个单词
}

int search(Trie *root,const char *s)  //查找某个单词是否已经存在
{
    Trie *p=root;
    //while(p!=NULL&&*s!='\0')
    while(p!=nullptr&&*s!='\0') // 或者使用NULL也可以，但nullptr需要使用c++11编译
    {
        p=p->next[*s-'a'];
        s++;
    }
    return (p!=NULL&&p->isWord==true);      //在单词结束处的标记为true时，单词才存在
}

void del(Trie *root)                      //释放整个字典树占的堆区空间
{
    int i;
    // 广度递归删除
    for(i=0;i<MAX;i++)
    {
        if(root->next[i]!=NULL)
        {
            del(root->next[i]);
        }
    }
    free(root);
}

int main(int argc, char *argv[])
{
    //int n,m;                              //n为建立Trie树输入的单词数，m为要查找的单词数
    //char s[100];
    std::cout<<"begin to process."<<endl;
    Trie *root= (Trie *)malloc(sizeof(Trie));
    init_trie(root);
    //scanf("%d",&n);
    //getchar();
    char dict_arr[][10] ={"hello","abc","hero","abandon","abort"};
    char query_arr[][10]={"hello","abo","hero","123"};
    //for(i=0;i<sizeof(char_arr)/sizeof(char*);i++)                 //先建立字典树
    std::cout<<"len: "<<sizeof(dict_arr)<<" single:"<<sizeof(dict_arr[0])<<endl;

    cout<<"dict:"<<endl;
    for(int i=0;i<5;i++)                 //先建立字典树
    {
        //scanf("%s",s);
        insert(root,dict_arr[i]);
        cout<<"insert: "<<dict_arr[i]<<" ";
    }

    char prefix[100]="\0";
    cout<<"\nbegin sort:"<<prefix<<endl;
    pre_order_visit(root,prefix,prefix);

    std::cout<<endl<<"begin to search!"<<endl;
    for(int i=0;i<4;i++)                 //查找
    {
        //scanf("%s",s);
        cout<<"query:"<<query_arr[i];
        if(search(root,query_arr[i])==1){
            cout<<" yes"<<endl;
        } else{
            cout<<" no"<<endl;
        }
    }
    del(root);                         //释放空间很重要
    return 0;
}