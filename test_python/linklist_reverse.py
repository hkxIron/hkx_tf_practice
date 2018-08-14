class Node():
    def __init__(self, val):
        self.val = val
        self.next = None

def reverse(head):
    if head is None or head.next is None:
        return head
    pre = None
    cur = head
    res = head
    while cur:
        res = cur
        tmp = cur.next
        cur.next = pre
        pre = cur
        cur = tmp
    return res

def visit_list(node):
    p= node
    while p:
        print(str(p.val) ,end=" ")
        p = p.next
"""
测试链表反转
"""
nums= [ 1,2,3,4,5 ]
head = Node(0)
last_node = head
for x in nums:
   node = Node(x)
   last_node.next = node
   last_node = node

visit_list(head)
reverse_head = reverse(head)
print("\n反转之后:")
visit_list(reverse_head)

