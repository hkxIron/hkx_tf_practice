// g++ -std=c++11 -o tree binary_tree.cc &&./tree.exe

#include<iostream>
#include<algorithm>
#include<iterator>
#include<string.h>
using namespace std;

struct tree{
    char data;
    struct tree* left; // 这里的struct关键字不可少
    struct tree* right;
};

tree* create_tree(char *& p_str ){
    if(*p_str=='#'||*p_str=='\0') {
        return NULL;
    }
    else{
        //tree* root = (tree*)malloc(sizeof(tree));
        tree* root = new tree();
        root->data=*(p_str++);
        root->left=create_tree(p_str);
        root->right=create_tree(++p_str); // 右子树时，指针必须要再次移动
        return root;
    }
}



void visit(tree* root){
    cout<<root->data<<" ";
}

void delete_tree(tree* root){
    cout<<"delete:"<<root->data<<"\n";
    delete root;
}

// 前序遍历
void pre_order_visit(tree* root,void(*fun)(tree*)){
    if(root==NULL) return;
    fun(root);
    pre_order_visit(root->left,fun);
    pre_order_visit(root->right,fun);
}

// 中序遍历
void in_order_visit(tree* root,void(*fun)(tree*)){
    if(root==NULL) return;
    in_order_visit(root->left,fun);
    fun(root);
    in_order_visit(root->right,fun);
}

// 后序遍历
void post_order_visit(tree* root,void(*fun)(tree*)){
    if(root==NULL) return;
    post_order_visit(root->left,fun);
    post_order_visit(root->right,fun);
    fun(root);
}

int main(){
    // 利用完全二叉树的数组表示，#代表空结点
    /*
          a
         /
        b
       / \
      c   d
         / \
        e   f
         \
          g
    */
    char a[100]="abc##de#g##f###";
    int len = strlen(a);
    cout<<"len:"<<len<<" size_of:"<<sizeof(a)<<endl;
    char* p_str=a;
    tree* root=create_tree(p_str);

    cout<<"pre order:"<<endl;
    pre_order_visit(root,visit);
    cout<<"\nin order:"<<endl;
    in_order_visit(root,visit);
    cout<<"\npost order:"<<endl;
    post_order_visit(root,visit);

    cout<<"\ndelete tree:"<<endl;
    post_order_visit(root,delete_tree);
    //--------------------
    //delete root;
    return 0;
}
