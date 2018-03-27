// g++ -std=c++11 -o tree binary_tree.cc &&./tree.exe

#include<algorithm>
#include<iostream>
#include<iterator>
#include<stack>
#include<string.h>
using namespace std;

struct tree{
    char data;
    struct tree* left; // 这里的struct关键字不可少
    struct tree* right;
};

// 从先序序列（含有#）中建立二叉树
tree* create_tree(char *& p_str ){
    if(*p_str=='#'||*p_str=='\0') {
        return NULL;
    }
    else{
        //tree* root = (tree*)malloc(sizeof(tree));
        tree* root = new tree();
        root->data=*(p_str);
        root->left=create_tree(++p_str);
        root->right=create_tree(++p_str); // 右子树时，指针必须要再次移动
        return root;
    }
}



void visit(tree* root){
    cout<<root->data<<" ";
}

void swap_child(tree* root){
    std::swap(root->left,root->right);
}

void delete_tree(tree* root){
    cout<<"delete:"<<root->data<<"\n";
    delete root;
}

// 前序遍历
void pre_order_visit(tree* root,void(*fun)(tree*)){ // 函数指针
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

// 中序非递归遍历
void in_order_non_recursive(tree *root){
    if(root == NULL) return;
    stack<tree*> st;
    tree*p=root;
    while(p||!st.empty()){
        if(p) { // 只会压入非空指针到栈内
            st.push(p);
            p = p->left;// 先将左边的全部入栈
        }
        else{
            p=st.top(); //top会返回元素
            st.pop(); // 注意:c++ pop时不会返回元素
            visit(p); // 元素出栈时立即访问
            p = p->right;
        }
    }
}


// 后序遍历
void post_order_visit(tree* root,void(*fun)(tree*)){
    if(root==NULL) return;
    post_order_visit(root->left,fun);
    post_order_visit(root->right,fun);
    fun(root);
}

// 二叉树的镜像
void post_mirror(tree* root){
    if(root==NULL) return;
    post_mirror(root->left);
    post_mirror(root->right);
    std::swap(root->left,root->right);
}

void pre_mirror(tree* root){
    if(root==NULL) return;
    std::swap(root->left,root->right);
    pre_mirror(root->left);
    pre_mirror(root->right);
}

int depth(tree* root){
    if(root==NULL) return 0;
    int left_depth=depth(root->left);
    int right_depth = depth(root->right);
    return std::max(left_depth,right_depth)+1;
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
    pre_order_visit(root,visit);// a b c d e g f
    cout<<"\nin order:"<<endl;
    in_order_visit(root,visit); // c b e g d f a
    cout<<"\npost order:"<<endl;
    post_order_visit(root,visit);// c g e f d b a

    // 二叉树镜像
    /*
                a
                 \
                  b
                 / \
                d   c
               / \
              f   e
                   \
                    g
    */
    // 镜像
    post_mirror(root);
    cout<<"\npre order after post mirror:"<<endl;
    pre_order_visit(root,visit);
    // 再镜像回来
    post_mirror(root);
    cout<<"\npre order after two mirror:"<<endl;
    pre_order_visit(root,visit);

    cout<<"\npre order after pre mirror:"<<endl;
    pre_mirror(root);
    pre_order_visit(root,visit);
    cout<<"\npre order after two mirror:"<<endl;
    pre_mirror(root);
    pre_order_visit(root,visit);
    cout<<"\npost order :"<<endl;
    post_order_visit(root,visit);

    cout<<"\npost mirror by visit:"<<endl;
    pre_order_visit(root,swap_child);
    pre_order_visit(root,visit);
    cout<<"\npre two mirror by visit:"<<endl;
    pre_order_visit(root,[](tree* root){ std::swap(root->left,root->right);}); // 匿名函数，不要捕获外部变量
    pre_order_visit(root,visit);
    //----树的深度
    cout<<"\ntree depth:"<<depth(root)<<endl;
    cout<<"\nin order non recursive visit:"<<endl;
    in_order_non_recursive(root);
    cout<<"\nin odrer recursive visit:"<<endl;
    in_order_visit(root,visit);
    //--------------------------
    cout<<"\ndelete tree:"<<endl;
    post_order_visit(root,delete_tree);
    return 0;
}
