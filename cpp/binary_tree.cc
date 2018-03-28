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


// 前序非递归遍历
/*
　根据前序遍历访问的顺序，优先访问根结点，然后再分别访问左孩子和右孩子。即对于任一结点，其可看做是根结点，因此可以直接访问，访问完之后，若其左孩子不为空，按相同规则访问它的左子树；当访问其左子树时，再访问它的右子树。因此其处理过程如下：

　　对于任一结点P：

     1)访问结点P，并将结点P入栈;

     2)判断结点P的左孩子是否为空，若为空，则取栈顶结点并进行出栈操作，并将栈顶结点的右孩子置为当前的结点P，循环至1);若不为空，则将P的左孩子置为当前的结点P;

     3)直到P为NULL并且栈为空，则遍历结束。

*/
void pre_order_non_recursive(tree* root){ // 该程序与 in_order_non_recursive2只有visit的位置不一样
    stack<tree*> s;
    tree *p=root;
    while(p||!s.empty()) {
        while(p){
            visit(p); // 先访问，再入栈
            s.push(p);
            p=p->left;
        }
        // p遇到了空指针
        if(!s.empty()) {
            p=s.top();
            s.pop();
            p=p->right;
        }
    }
}

// 中序非递归遍历
/*
　根据中序遍历的顺序，对于任一结点，优先访问其左孩子，而左孩子结点又可以看做一根结点，然后继续访问其左孩子结点，直到遇到左孩子结点为空的结点才进行访问，然后按相同的规则访问其右子树。因此其处理过程如下：

　　对于任一结点P，

  　1)若其左孩子不为空，则将P入栈并将P的左孩子置为当前的P，然后对当前结点P再进行相同的处理；

 　 2)若其左孩子为空，则取栈顶元素并进行出栈操作，访问该栈顶结点，然后将当前的P置为栈顶结点的右孩子；

  　3)直到P为NULL并且栈为空则遍历结束。

*/
void in_order_non_recursive2(tree *root){
    stack<tree*> s;
    tree *p=root;
    while(p||!s.empty())
    {
        while(p) {
            s.push(p);
            p=p->left; // 向左走到底
        }
        if(!s.empty()) { //取出栈顶并访问
            p=s.top();
            visit(p);
            s.pop();
            p=p->right;
        }
    }
}

// 中序非递归遍历
void in_order_non_recursive(tree *root){
    if(root == NULL) return;
    stack<tree*> st;
    tree*p=root;
    while(p||!st.empty()){
        if(p) { // 只会压入非空指针到栈内
            st.push(p);
            p = p->left;// 先将左边的入栈
        }
        else{
            p=st.top(); //top会返回元素
            st.pop(); // 注意:c++ pop时不会返回元素
            visit(p); // 元素出栈时立即访问
            p = p->right;
        }
    }
}

//后序非递归遍历
/* 第二种思路：要保证根结点在左孩子和右孩子访问之后才能访问，因此对于任一结点P，先将其入栈。
如果P不存在左孩子和右孩子，则可以直接访问它；
或者P存在左孩子或者右孩子，但是其左孩子和右孩子都已被访问过了，
则同样可以直接访问该结点。

若非上述两种情况，则将P的右孩子和左孩子依次入栈，这样就保证了 每次取栈顶元素的时候，
左孩子在右孩子前面被访问，左孩子和右孩子都在根结点前面被访问。
*/
void post_order_non_recursive(tree*root){
    stack<tree*> s;
    tree *cur;                      //当前结点
    tree *pre=NULL;                 //前一次访问的结点
    s.push(root); // 只有后序非递归遍历需要入栈
    while(!s.empty()) {
        cur=s.top(); // 只取元素，但并未出栈
        if((cur->left==NULL&&cur->right==NULL)|| // P不存在左孩子和右孩子，则可以直接访问它
           (pre!=NULL&&(pre==cur->left||pre==cur->right))) // P存在左孩子或者右孩子，但是其左孩子和右孩子都已被访问过了
        {
            visit(cur);  //如果当前结点没有孩子结点或者孩子节点都已被访问过
            s.pop(); // 出栈只有一个地方
            pre=cur;
        }
        else {
            //注意，二者顺序不可改变
            // P的右孩子和左孩子依次入栈，这样就保证了 每次取栈顶元素的时候，左孩子在右孩子前面被访问，左孩子和右孩子都在根结点前面被访问
            if(cur->right!=NULL) s.push(cur->right);
            if(cur->left!=NULL) s.push(cur->left);
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

// 求树的深度
int depth(tree* root){
    if(root==NULL) return 0;
    int left_depth=depth(root->left);
    int right_depth = depth(root->right);
    return std::max(left_depth,right_depth)+1;
}

// 最近公共祖先
tree* lowest_common_ancestor(tree* root, tree* p, tree* q) {
    if(root==NULL||root==p||root==q) return root;
    tree* left = lowest_common_ancestor(root->left,p,q);
    tree* right =lowest_common_ancestor(root->right,p,q);
    if(left!=NULL&&right!=NULL) return root; // 当前节点为公共,两边的子节点里包含p和q
    return left?left:right;
}

void display(tree *root)        //显示树形结构
 {
     if(root!=NULL)
     {
         cout<<root->data;
         if(root->left!=NULL)
         {
             cout<<'(';
             display(root->right);
         }
         if(root->right!=NULL)
         {
             cout<<',';
             display(root->right);
             cout<<')';
         }
     }
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
    cout<<"\nin order recursive visit:"<<endl;
    in_order_visit(root,visit);
    // 前序非递归
    cout<<"\npre order non recursive visit:"<<endl;
    pre_order_non_recursive(root);
    // 中序非递归
    cout<<"\nin order non recursive visit:"<<endl;
    in_order_non_recursive(root);
    cout<<"\nin order non recursive visit2:"<<endl;
    in_order_non_recursive2(root);
    // 后序非递归
    cout<<"\npost order non recursive visit:"<<endl;
    post_order_non_recursive(root);
    //--------------------------
    cout<<"\ndisplay:"<<endl;
    display(root);
    cout<<"\ndelete tree:"<<endl;
    post_order_visit(root,delete_tree);
    return 0;
}
