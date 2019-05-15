// g++ -std=c++11 -o tree binary_tree.cc &&./tree
// blog:https://github.com/hkxIron/algorithm/blob/master/sword_offer/src/023.cpp

#include<algorithm>
#include<iostream>
#include<iterator>
#include<string.h>
#include<stack>
#include<vector>
#include<queue>
using namespace std;

// 注意:此处可以不用 typedef
struct tree{
    char data;
    struct tree* left; // 这里的struct关键字不可少
    struct tree* right;
};

// 从先序序列（含有#）建立二叉树
// str = "abc##de#g##f###";
tree* create_tree(char *& p_str ){
    if(*p_str=='#'||*p_str=='\0') {
        return NULL;
    }
    else{
        //tree* root = (tree*)malloc(sizeof(tree));
        tree* root = new tree();
        root->data=*(p_str); // 将原来字符串的值赋给data
        root->left=create_tree(++p_str);
        root->right=create_tree(++p_str); // 右子树时，指针必须要再次移动
        return root;
    }
}

void visit(tree* root){
    cout<<root->data<<" ";
}

void swap_child(tree* root){
    std::swap(root->left, root->right);
}

void delete_tree(tree* root){
    cout<<"delete:"<<root->data<<"\n";
    delete root;
}

// 前序遍历
void pre_order_visit(tree* root, void(*fun)(tree*)){ // 函数指针
    if(root==NULL) return;
    fun(root);
    pre_order_visit(root->left, fun);
    pre_order_visit(root->right, fun);
}

// 中序遍历
void in_order_visit(tree* root, void(*fun)(tree*)){
    if(root==NULL) return;
    in_order_visit(root->left, fun);
    fun(root);
    in_order_visit(root->right, fun);
}

// 结论:如果在算法中暂且抹去和递归无关的Visit语句, 则3个遍历算法完全相同。

// 前序非递归遍历
/*
　根据前序遍历访问的顺序，优先访问根结点，然后再分别访问左孩子和右孩子。即对于任一结点，其可看做是根结点，因此可以直接访问，
    访问完之后，若其左孩子不为空，按相同规则访问它的左子树；当访问其左子树时，再访问它的右子树。因此其处理过程如下：
　　对于任一结点P：
     1)访问结点P，并将结点P入栈;
     2)判断结点P的左孩子是否为空，若为空，则取栈顶结点并进行出栈操作，并将栈顶结点的右孩子置为当前的结点P，循环至1);
        若不为空，则将P的左孩子置为当前的结点P;
     3)直到P为NULL并且栈为空，则遍历结束。
*/
void pre_order_non_recursive(tree* root){ // 该程序与 in_order_non_recursive2只有visit的位置不一样
    stack<tree*> s;
    tree *p=root;
    while(p||!s.empty()) {
        while(p) {
            visit(p); // 左子树while访问到底, 先访问，再入栈
            s.push(p);
            p=p->left;
        }
        // p向右访问遇到了空指针, 弹出元素，并转向他的右孩子
        if(!s.empty()) {
            p=s.top(); // 取(但并未弹出)栈顶元素
            s.pop(); // 弹出
            p=p->right;
        }
    }
}

// 中序非递归遍历
/*
　根据中序遍历的顺序，对于任一结点，优先访问其左孩子，而左孩子结点又可以看成根结点，然后继续访问其左孩子结点，
  直到遇到左孩子结点为空的结点才进行访问，然后按相同的规则访问其右子树。因此其处理过程如下：

　　对于任一结点P，
  　1)若其左孩子不为空，则将P入栈并将P的左孩子置为当前的P，然后对当前结点P再进行相同的处理；
 　 2)若其左孩子为空，则取栈顶元素并进行出栈操作，访问该栈顶结点，然后将当前的P置为栈顶结点的右孩子；
  　3)直到P为NULL并且栈为空则遍历结束。
*/
// 推荐此种非递归中序遍历: 左根右
void in_order_non_recursive2(tree *root){
    stack<tree*> s;
    tree *p=root;
    while(p||!s.empty()) {
        while(p) {
            s.push(p);
            p=p->left; // 向左走到底
        }
        if (!s.empty()) { // 取出栈顶并访问后，再转向右孩子，（即只在出栈时才访问）
            p=s.top(); // 只取出栈顶元素，但并不出栈该元素
            visit(p); // 出栈时才访问
            s.pop(); //弹出
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
        else{ // 左子树遇到空指针
            p=st.top(); //top会返回元素
            st.pop(); // 注意:c++ pop时不会返回元素
            visit(p); // 元素出栈时立即访问
            p = p->right; //需要转右子树
        }
    }
}

// 后序非递归遍历
/* 第二种思路：
要保证根结点在左孩子和右孩子访问之后才能访问，因此对于任一结点P，先将其入栈。
如果P不存在左孩子和右孩子，则可以直接访问它；
或者P存在左孩子或者右孩子，但是其左孩子和右孩子都已被访问过了，
则同样可以直接访问该结点。

若非上述两种情况，则将P的右孩子和左孩子依次入栈，这样就保证了 每次取栈顶元素的时候，
左孩子在右孩子前面被访问，左孩子和右孩子都在根结点前面被访问。

[[ 注意：后序遍历时，栈中存储的都是当前节点的祖先。即递归遍历时，栈中存储的是祖先节点。]]
*/
void post_order_non_recursive(tree*root){
    stack<tree*> s;
    tree *cur;                      // 当前结点
    tree *pre=NULL;                 // 需要存储前一次访问的结点
    if(root!=NULL) s.push(root); // 只有后序非递归遍历需要事先入栈(不能先访问,而是先入栈)
    while(!s.empty()) {
        cur = s.top(); // 只取元素，但并未出栈
        if((cur->left==NULL&&cur->right==NULL)|| // P不存在左孩子和右孩子，则可以直接访问它
           (pre!=NULL&&(pre==cur->left||pre==cur->right))) //或 P存在左孩子或者右孩子，但是其左孩子和右孩子都已被访问过了（前一次访问的是它的左右孩子）
        {
            visit(cur);  //如果当前结点没有孩子结点或者孩子节点都已被访问过
            s.pop(); // 出栈只有一个地方
            pre=cur;
        } else {
            // P的右孩子和左孩子依次入栈，这样就保证了每次取栈顶元素的时候，
            // 左孩子在右孩子前面被访问，左孩子和右孩子都在根结点前面被访问
            // 注意，二者顺序不可改变
            if(cur->right!=NULL) s.push(cur->right); // 先存右孩子,保证访问时 先出并访问左孩子
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
    // 后序遍历
    post_mirror(root->left);
    post_mirror(root->right);
    std::swap(root->left,root->right);
}

// 二叉树镜像(前序遍历)
void pre_mirror(tree* root){
    if(root==NULL) return;
    std::swap(root->left, root->right);
    pre_mirror(root->left);
    pre_mirror(root->right);
}

// 求树的深度
int depth(tree* root){
    if(root==NULL) return 0;
    int left_depth=depth(root->left);
    int right_depth = depth(root->right);
    // 因为当前结点不为空，那么 深度是子树 + 1
    return std::max(left_depth, right_depth) + 1;
}

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
// 普通树的求公共结点,请见 ../CommonParentInTree
/*
如果只是一般的二叉树，不是搜索树的话 可以用递归的方法,
方法思路如下：首先看root是否为公共祖先，不是的话，则递归到左右节点（可能效率不怎么好）
*/
// 求p,q的最近公共祖先
// root:为根节点， p,q为两个输入的最近公共祖先
tree* lowest_common_ancestor(tree* root, tree* p, tree* q) {
    if(root==NULL||root==p||root==q) return root; // 返回p或者q或者null
    tree* left = lowest_common_ancestor(root->left, p, q); // 它们返回时，返回的是输入的p或者q或者null
    tree* right = lowest_common_ancestor(root->right, p, q);

    // 当前节点为公共,因为两边的子节点里包含p和q, 只有当左右孩子都不为空时，才返回当前结点
    if(left!=NULL&&right!=NULL) return root;
    return left?left:right; // 否则，返回两边非空的那边, 比如从g返回到结点e
}

/**

 求p,q的最近公共祖先
 二叉排序树:
            5
           / \
          3   8
         /\  / \
        1 4 6  9
       / \
      0  2

 1.如果树是搜索二叉树(即排序二叉树, 位于左子树的节点都比父节点小点,位于右子树的节点都比节点大),
     我们只需要从树的根节点r开始与两个输入节点p,q进行比较.
     如果p,q都小于r,那么最低的公共祖先在r的左子树中,
     如果p,q都大于r,那么最低的公共祖先在r的右子树中,
     如果r介于p,q之间,那么当前r就是最低公共祖先

 2.如果不是二叉树,只是普通的树
    问是二叉树中的结点是否有指向父结点的指针
    如果有指向父结点的指针,那么这些链表的尾指针都是指向根结点,
    它们的最低公共祖先就是这两个链表的第一个公共节点,因此转化为求两个相交链表的公共节点,比较简单

 3.如果只是普通的树,且没有指向父结点的指针
    遍历时将父结点存储起来, 然后求公共结点.

*/

/*
	[对称的二叉树]

    [题目]
	请实现一个函数，用来判断一颗二叉树是不是对称的。注意，如果一个二叉树同此二叉树的镜像是同样的，定义其为对称的。
    [解析]
    可以做图抽象便于理解。注意边界条件。

    思路：
想一下打印输出某二叉树的镜像，实现的思路是：采用层序遍历的思路对每一个遍历的节点，如果其有孩子节点，
那么就交换两者。直到遍历的节点没有孩子节点为止，然而此题是对二叉树镜像的判断，
明显是更简单的，只需要进行两个判断：对节点的左孩子与其兄弟节点右孩子的判断以及对节点右孩子与其兄弟节点左孩子的判断。
这样就完成对对一棵二叉树是否对称的判断.

         8
       /  \
      6    6
     / \  / \
    5  7 7   5


*/

bool is_symmetrical(tree* p_left,tree* p_right){
    if(p_left==NULL&&p_right==NULL) return true;// 都为空，肯定是对称的
    if(p_left==NULL||p_right==NULL) return false; // 有一个不为空，肯定不对称
    if(p_left->data!=p_right->data) return false; // 数据不相等，也不对称
    return is_symmetrical(p_left->left,p_right->right) // 左边的左孩子与右边的右孩子对称
        &&is_symmetrical(p_left->right,p_right->left); // 左边的右孩子与右边的左孩子对称
}

bool is_symmetrical(tree* root){
    if(root==NULL) return true;
    return is_symmetrical(root->left,root->right);
}

/* z字型遍历二叉树,如：
         8
       /  \
      6    9
      \   / \
      7  4   5
其序列顺序为:
 8 
 9 6 
 7 4 5
思想是层次遍历，加上方向判断

queue的用法：
queue与stack模版非常类似，queue模版也需要定义两个模版参数，一个是元素类型，一个是容器类型，元素类型是必要的，容器类型是可选的，默认为dqueue类型。

定义queue对象的示例代码如下：
queue<int>q1;
queue<double>q2;
queue的基本操作有：
1.入队：如q.push(x):将x元素接到队列的末端；
2.出队：如q.pop() 弹出队列的第一个元素，并不会返回元素的值；
3,访问队首元素：如q.front()
4,访问队尾元素，如q.back();
5,访问队中的元素个数，如q.size();

由于需要前插，此处使用vector

*/ 

// 层次遍历
void level_visit_tree(tree* root, void(*fun)(tree *)) {
  tree*p = root;
  if(p == nullptr) return;
  vector<tree*> que;
  que.push_back(p);   
  int level = 0;
  while(!que.empty()){
    int count_in_level = que.size(); //本层需要遍历节点的个数
    for(int i=0;i<count_in_level;i++){
        p = que.front(); // 获取队首
        fun(p); // 访问当前节点
        que.erase(que.begin(), que.begin()+1); // 弹出队首
        if(p->left) que.push_back(p->left); //后插
        if(p->right) que.push_back(p->right);
    }
    cout<<endl;
    level++;
  }
}

// TODO: 好像有bug,但并未找到bug在哪里
void zig_zag_visit_tree(tree* root, void(*fun)(tree *)) {
  tree*p = root;
  if(p == nullptr) return;
  vector<tree*> que;
  que.push_back(p);   
  int level = 0;
  while(!que.empty()){
    int count_in_level = que.size(); //本层需要遍历节点的个数
    cout<<"level count:"<<count_in_level<<endl;
    for(int i=0;i<count_in_level;i++){
        p = que.front(); // 获取队首
        fun(p); // 访问当前节点
        que.erase(que.begin(), que.begin()+1); // 弹出队首
        int num_of_last_level = count_in_level - i - 1;
        if(level%2==0){
            //if(p->left) que.push_back(p->left); //后插
            //if(p->right) que.push_back(p->right);
            //不管是否逆序,都需要前插,达到类似于stack的效果
            if(p->left) que.insert(que.begin()+num_of_last_level, p->left); // 前插时，要在上一层之后的元素后面开始插入
            if(p->right) que.insert(que.begin()+num_of_last_level, p->right);
        }else{
            // 当上一层是逆序时,本层需要从右孩子前插
            if(p->right) que.insert(que.begin()+num_of_last_level, p->right);
            if(p->left) que.insert(que.begin()+num_of_last_level, p->left); // 前插时，要在上一层之后的元素后面开始插入
        }
    }
    level++;
    cout<<endl;
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
    char a[100]="abc##de#g##f###"; //从先序序列（含有#）中建立二叉树 
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
    //最近公共祖先
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
    tree* p_c = root->left->left;
    tree* p_g = root->left->right->left->right;
    tree * p_common = lowest_common_ancestor(root, p_c, p_g);
    cout<<"\ncommon ancesor1:"<<p_common->data<<endl; // b
    tree* p_a = root;
    tree * p_common2 = lowest_common_ancestor(root, p_a, p_g);
    cout<<"\ncommon ancesor2:"<<p_common2->data<<endl; // a
    //--------------------------
    cout<<"\ndelete tree:"<<endl;
    post_order_visit(root,delete_tree);
    //----------------
    {
        /*
              a
             / \
            b   j
           / \
          c   d
         /\   / \
         h m  e   f
        /   /  \  \
       k   i    g  l
        
        */
        char a[100]="abchk###m##dei##g##f#l##j##"; //从先序序列（含有#）中建立二叉树 
        char* p_str=a;
        tree* root=create_tree(p_str);
        cout<<"\n层次访问二叉树:"<<endl;
        level_visit_tree(root, visit);
        cout<<"\nzig zag访问二叉树:"<<endl;
        zig_zag_visit_tree(root, visit);
        cout<<"\npost order tree:"<<endl;
        post_order_visit(root,visit);
        cout<<"\ndelete tree:"<<endl;
        post_order_visit(root,delete_tree);
    }
    return 0;
}
