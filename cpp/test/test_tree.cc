#include<iostream>
#include<string>
using namespace std;
struct BNode{//二叉树节点
    BNode(const char d='#'):data(d), left(nullptr), right(nullptr) {};
    char data;
    BNode* left;
    BNode* right;
};
//根据先序遍历构建一棵二叉树，返回root指针
BNode* constructBinaryTree(const string& preOrder, unsigned& index){
    if (preOrder.size() == 0 || index == preOrder.size() || preOrder[index] == '#')//若空串或者index超出范围，则返回空指针
        return nullptr;
    BNode* T = new BNode(preOrder[index++]);
    T->left = constructBinaryTree(preOrder, index);
    T->right = constructBinaryTree(preOrder, ++index);
    return T;
}
void preOrder(BNode* T){
    if (T != nullptr){
        cout << T->data << " ";
        preOrder(T->left);
        preOrder(T->right);
    }
}

void postOrder(BNode* T){
    if (T != nullptr){
        postOrder(T->left);
        postOrder(T->right);
        cout << T->data << " ";
    }
}
//中序遍历
void mediaOrder(BNode* T){
    if (T != nullptr){
        mediaOrder(T->left);
        cout << T->data << " ";
        mediaOrder(T->right);
    }
}


int main(){
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

    string str="abc##de#g##f###";
    unsigned index = 0;
    BNode* root = constructBinaryTree(str, index);
    cout<<"pre order:";
    preOrder(root);
    cout<< "\n in order ";
    mediaOrder(root);
    cout<< "\n post order ";
    postOrder(root);
    return 0;
}