import time
from timeit import Timer

"""
如果 a+b+c=1000，且 a^2+b^2=c^2（a,b,c 为自然数），如何求出所有a、b、c可能的组合?
"""


def test1():
    start = time.time()
    for a in range(0, 1001):
        for b in range(0, 1001 - a):
            c = 1000 - a - b
            if a ** 2 + b ** 2 == c ** 2:
                print(a, b, c)
    end = time.time()
    print(end - start)


"""
“大O记法”：对于单调的整数函数f，如果存在一个整数函数g和实常数c>0，使得对于充分大的n总有f(n)<=c*g(n)，就说函数g是f的一个渐近函数（忽略常数），记为f(n)=O(g(n))。也就是说，在趋向无穷的极限意义下，函数f的增长速度受到函数g的约束，亦即函数f与函数g的特征相似。

时间复杂度：假设存在函数g，使得算法A处理规模为n的问题示例所用时间为T(n)=O(g(n))，则称O(g(n))为算法A的渐近时间复杂度，简称时间复杂度，记为T(n)
"""

"""
基本操作，即只有常数项，认为其时间复杂度为O(1)
顺序结构，时间复杂度按加法进行计算
循环结构，时间复杂度按乘法进行计算
分支结构，时间复杂度取最大值
判断一个算法的效率时，往往只需要关注操作数量的最高次项，其它次要项和常数项可以忽略
在没有特殊说明时，我们所分析的算法的时间复杂度都是指最坏时间复杂度
"""

"""
顺序表

1.元素大小相同
元素存储的物理地址（实际内存地址）可以通过存储区的起始地址Loc (e0)加上逻辑地址（第i个元素）与存储单元大小（c）的乘积计算而得
Loc(ei) = Loc(e0) + c*i

2.元素的大小不同
则需采用元素外置的形式，将实际数据元素另行存储，而顺序表中各单元位置保存对应元素的地址信息（即链接）


顺序表的两种基本实现方式
1.一体式结构，存储表信息的单元与元素存储区以连续的方式安排在一块存储区里，两部分数据的整体形成一个完整的顺序表对象
2.分离式结构，表对象里只保存与整个表有关的信息（即容量和元素个数），实际数据元素存放在另一个独立的元素存储区里，通过链接与基本表对象关联

顺序表的操作
1. 尾端加入/删除元素，时间复杂度为O(1)
2. 非保序的加入/删除元素（不常见），时间复杂度为O(1)
3. 保序的元素加入/删除，时间复杂度为O(n)
"""

"""
链表

链表（Linked list）是一种常见的基础数据结构，是一种线性表，但是不像顺序表一样连续存储数据，而是在每一个节点（数据存储单元）里存放下一个节点的位置信息（即地址）

注意虽然表面看起来复杂度都是 O(n)，但是链表和顺序表在插入和删除时进行的是完全不同的操作。链表的主要耗时操作是遍历查找，删除和插入操作本身的复杂度是O(1)。顺序表查找很快，主要耗时的操作是拷贝覆盖。因为除了目标元素在尾部的特殊情况，顺序表进行插入和删除时需要对操作点之后的所有元素进行前后移位操作，只能通过拷贝和覆盖的方法进行
"""


class SingleNode(object):
    """单链表的结点"""

    def __init__(self, item):
        # item存放数据元素
        self.item = item
        # next是下一个节点的标识
        self.next = None


class SingLinkList:
    """
    单链表
    """

    def __init__(self):
        self.__head = None

    def is_empty(self):
        """判断链表是否为空"""
        return self.__head is None

    def length(self):
        """链表长度"""
        count = 0
        cursor = self.__head
        while cursor is not None:
            count += 1
            cursor = cursor.next
        return count

    def travel(self):
        """遍历链表"""
        cursor = self.__head
        while cursor is not None:
            print(cursor.item)
            cursor = cursor.next
        print("")

    def add(self, item):
        """头部添加元素  O(1)"""
        node = SingleNode(item)
        node.next = self.__head
        self.__head = node

    def append(self, item):
        """尾部添加元素  O(n)"""
        node = SingleNode(item)
        if self.__head is None:
            self.__head = node
        else:
            cursor = self.__head
            while cursor.next is not None:
                cursor = cursor.next
            cursor.next = node

    def insert(self, index, item):
        """指定位置添加元素  O(n)"""
        if index <= 0:
            self.add(item)
        elif index > (self.length() - 1):
            self.append(item)
        else:
            count = 0
            cursor = self.__head
            node = SingleNode(item)
            while cursor is not None:
                if count == (index - 1):
                    node.next = cursor.next
                    cursor.next = node
                    break
                count += 1
                cursor = cursor.next

    def remove(self, item):
        """删除节点"""
        cursor = self.__head
        previous = None
        while cursor is not None:
            if cursor.item == item:
                if previous is None:
                    # 如果第一个节点就是需要删除的节点
                    self.__head = cursor.next
                else:
                    previous.next = cursor.next
                break
            else:
                previous = cursor
                cursor = cursor.next

    def search(self, item):
        """链表查找节点是否存在，并返回True或者False  O(n)"""
        cursor = self.__head
        while cursor is not None:
            if item == cursor.item:
                return True
            cursor = cursor.next
        return False


class DoubleNode:

    def __init__(self, item):
        self.item = item
        self.next = None
        self.prev = None


class DoubleLinkList:

    def __init__(self):
        self.__head = None

    def is_empty(self):
        return self.__head is None

    def length(self):
        count = 0
        cursor = self.__head
        while cursor is not None:
            count += 1
            cursor = cursor.next
        return count

    def travel(self):
        cursor = self.__head
        while cursor is not None:
            print(cursor.item)
            cursor = cursor.next
        print("")

    def add(self, item):
        node = DoubleNode(item)
        node.next = self.__head

        if self.is_empty():
            self.__head = node
        else:
            self.__head.prev = node
            self.__head = node

    def append(self, item):
        node = DoubleNode(item)

        if self.is_empty():
            self.__head = node
        else:
            cursor = self.__head
            while cursor.next is not None:
                cursor = cursor.next

            cursor.next = node
            node.prev = cursor

    def insert(self, index, item):
        if index <= 0:
            self.add(item)
        elif index > (self.length() - 1):
            self.append(item)
        else:
            count = 0
            cursor = self.__head
            node = DoubleNode(item)
            while cursor is not None:
                if index == count:
                    cursor.prev.next = node
                    cursor.prev = node
                    node.prev = cursor.prev
                    node.next = cursor
                    break

                cursor = cursor.next
                count += 1

    def remove(self, item):
        cursor = self.__head
        while cursor is not None:
            if cursor == item:
                if cursor == self.__head:
                    self.__head = cursor.next
                    if cursor.next:
                        cursor.next.prev = None
                else:
                    cursor.prev.next = cursor.next
                    if cursor.next:
                        cursor.next.prev = cursor.prev
                break

            else:
                cursor = cursor.next


"""
栈（stack），有些地方称为堆栈，是一种容器，可存入数据元素、访问元素、删除元素，它的特点在于只能允许在容器的一端（称为栈顶端指标，英语：top）进行加入数据（英语：push）和输出数据（英语：pop）的运算。没有了位置概念，保证任何时候可以访问、删除的元素都是此前最后存入的那个元素，确定了一种默认的访问顺序。

由于栈数据结构只允许在一端进行操作，因而按照后进先出（LIFO, Last In First Out）的原理运作。

栈可以用顺序表实现，也可以用链表实现。
"""

"""
队列（queue）是只允许在一端进行插入操作，而在另一端进行删除操作的线性表。

队列是一种先进先出的（First In First Out）的线性表，简称FIFO

同栈一样，队列也可以用顺序表或者链表实现。
"""


def bubble_sort(a_list):
    """
    冒泡排序

    最优时间复杂度：O(n) （表示遍历一次发现没有任何可以交换的元素，排序结束。）
    最坏时间复杂度：O(n2)
    稳定性：稳定
    """
    # 每进行一次循环比较，下一次需要进行比较的次数就减1
    for i in range(len(a_list) - 1, 0, -1):
        for j in range(i):
            if a_list[j] > a_list[j + 1]:
                a_list[j], a_list[j + 1] = a_list[j + 1], a_list[j]
    return a_list


def selection_sort(a_list):
    """
    选择排序（Selection sort）是一种简单直观的排序算法。它的工作原理如下。首先在未排序序列中找到最小（大）元素，存放到排序序列的起始位置，然后，再从剩余未排序元素中继续寻找最小（大）元素，然后放到已排序序列的末尾。以此类推，直到所有元素均排序完毕。

    最优时间复杂度：O(n2)
    最坏时间复杂度：O(n2)
    稳定性：不稳定（考虑升序每次选择最大的情况）
    """
    for i in range(len(a_list) - 1):
        # 首先假设当前的数为最小数
        min_index = i

        # 遍历当前数之后的所有数字，判断大小
        for j in range(i + 1, len(a_list)):
            if a_list[j] < a_list[min_index]:
                min_index = j

        # 如果遍历完成后，发现最小数不是当前数，则将当前数和最小数调换位置
        if min_index != i:
            a_list[i], a_list[min_index] = a_list[min_index], a_list[i]
    return a_list


def insertion_sort1(a_list):
    sorted_list = [a_list[0]]
    # for i in range(0, len(a_list) - 1):
    for j in range(1, len(a_list)):
        max_index = len(sorted_list)
        for x in range(len(sorted_list), 0, -1):
            if a_list[j] < sorted_list[x - 1]:
                max_index = x - 1
            else:
                break
        if max_index == len(sorted_list):
            sorted_list.append(a_list[j])
        else:
            sorted_list.insert(max_index, a_list[j])
    return sorted_list


def insertion_sort2(alist):
    """
    插入排序（英语：Insertion Sort）是一种简单直观的排序算法。它的工作原理是通过构建有序序列，对于未排序数据，在已排序序列中从后向前扫描，找到相应位置并插入。插入排序在实现上，在从后向前扫描过程中，需要反复把已排序元素逐步向后挪位，为最新元素提供插入空间。

    最优时间复杂度：O(n) （升序排列，序列已经处于升序状态）
    最坏时间复杂度：O(n2)
    稳定性：稳定
    """
    # 从第二个位置，即下标为1的元素开始向前插入
    for i in range(1, len(alist)):
        # 从第i个元素开始向前比较，如果小于前一个元素，交换位置
        for j in range(i, 0, -1):
            if alist[j] < alist[j - 1]:
                alist[j], alist[j - 1] = alist[j - 1], alist[j]
    return alist


def quick_sort(alist, start, end):
    """
    快速排序

    快速排序（英语：Quicksort），又称划分交换排序（partition-exchange sort），通过一趟排序将要排序的数据分割成独立的两部分，其中一部分的所有数据都比另外一部分的所有数据都要小，然后再按此方法对这两部分数据分别进行快速排序，整个排序过程可以递归进行，以此达到整个数据变成有序序列。

    理解：默认将一个数组的第一个数字作为低位low，最后一个数字作为高位high， 并以第一个数字作为基准mid。首先移动高位high游标，如果其值大于基准mid，则向下一位(左)移动，否则说明其值应该在基准左边，则将其值赋予此时的低位low。接下来移动低位low游标，如果其值小于基准mid，则向下一位(右)移动，否则说明其值在基准右边，则将其值赋予此时的高位high。当low和high重合，则当前low(high)位就是基准位，因为比它小的都已经移动到了它左边，比它大的都移动到了它右边，接下来只需要递归地对它左边、右边的数字进行上述操作即可

    最优时间复杂度：O(nlogn)
    最坏时间复杂度：O(n2)
    稳定性：不稳定
    """

    # 递归的退出条件
    if start >= end:
        return

    # 设定起始元素为要寻找位置的基准元素
    mid = alist[start]

    # low为序列左边的由左向右移动的游标
    low = start

    # high为序列右边的由右向左移动的游标
    high = end

    while low < high:
        # 如果low与high未重合，high指向的元素不比基准元素小，则high向左移动
        while low < high and alist[high] >= mid:
            high -= 1
        # 将high指向的元素放到low的位置上
        alist[low] = alist[high]

        # 如果low与high未重合，low指向的元素比基准元素小，则low向右移动
        while low < high and alist[low] < mid:
            low += 1
        # 将low指向的元素放到high的位置上
        alist[high] = alist[low]

    # 退出循环后，low与high重合，此时所指位置为基准元素的正确位置
    # 将基准元素放到该位置
    alist[low] = mid

    # 对基准元素左边的子序列进行快速排序
    quick_sort(alist, start, low - 1)

    # 对基准元素右边的子序列进行快速排序
    quick_sort(alist, low + 1, end)


def shell_sort(alist):
    """
    希尔排序

    希尔排序是把记录按下标的一定增量分组，对每组使用直接插入排序算法排序；随着增量逐渐减少，每组包含的关键词越来越多，当增量减至1时，整个文件恰被分成一组，算法便终止。

    最优时间复杂度：根据步长序列的不同而不同
    最坏时间复杂度：O(n2)
    稳定性：不稳定
    """
    n = len(alist)
    # 初始步长
    gap = int(n / 2)
    while gap > 0:
        # 按步长进行插入排序
        for i in range(gap, n):
            j = i
            # 插入排序
            while j >= gap and alist[j - gap] > alist[j]:
                alist[j - gap], alist[j] = alist[j], alist[j - gap]
                j -= gap
        # 得到新的步长
        gap = int(gap / 2)
    return alist


def merge_sort(alist):
    """
    归并排序是采用分治法的一个非常典型的应用。归并排序的思想就是先递归分解数组，再合并数组。

    将数组分解最小之后，然后合并两个有序数组，基本思路是比较两个数组的最前面的数，谁小就先取谁，取了后相应的指针就往后移一位。然后再比较，直至一个数组为空，最后把另一个数组的剩余部分复制过来即可。
    """
    if len(alist) <= 1:
        return alist
    # 二分分解
    num = int(len(alist)/2)
    left = merge_sort(alist[:num])
    right = merge_sort(alist[num:])
    # 合并
    return merge(left, right)


def merge(left, right):
    """
    合并操作，将两个有序数组left[]和right[]合并成一个大的有序数组
    """

    # left与right的下标指针
    l, r = 0, 0
    result = []
    while l < len(left) and r < len(right):
        if left[l] <= right[r]:
            result.append(left[l])
            l += 1
        else:
            result.append(right[r])
            r += 1
    result += left[l:]
    result += right[r:]
    return result


def binary_search(a_list, item):
    """
    搜索的几种常见方法：顺序查找、二分法查找、二叉树查找、哈希查找

    --- 二分法查找 ---
    二分查找又称折半查找，优点是比较次数少，查找速度快，平均性能好；其缺点是要求待查表为有序表，且插入删除困难
    因此，折半查找方法适用于不经常变动而查找频繁的有序列表

    首先，假设表中元素是按升序排列，将表中间位置记录的关键字与查找关键字比较，如果两者相等，则查找成功；否则利用中间位置记录将表分成前、后两个子表，如果中间位置记录的关键字大于查找关键字，则进一步查找前一子表，否则进一步查找后一子表。

    最优时间复杂度：O(1)
    最坏时间复杂度：O(logn)
    """
    mid = int(len(a_list) / 2)
    mid_value = a_list[mid]
    if mid_value > item:
        return binary_search(a_list[:mid], item)
    elif mid_value < item:
        return mid + 1 + binary_search(a_list[(mid+1):], item)
    else:
        return mid


"""
树（英语：tree）是一种抽象数据类型（ADT）或是实作这种抽象数据类型的数据结构，用来模拟具有树状结构性质的数据集合。它是由n（n>=1）个有限节点组成一个具有层次关系的集合。

节点的度：一个节点含有的子树的个数称为该节点的度；
树的度：一棵树中，最大的节点的度称为树的度；
叶节点或终端节点：度为零的节点；
父亲节点或父节点：若一个节点含有子节点，则这个节点称为其子节点的父节点；
孩子节点或子节点：一个节点含有的子树的根节点称为该节点的子节点；
兄弟节点：具有相同父节点的节点互称为兄弟节点；
节点的层次：从根开始定义起，根为第1层，根的子节点为第2层，以此类推；
树的高度或深度：树中节点的最大层次；
堂兄弟节点：父节点在同一层的节点互为堂兄弟；
节点的祖先：从根到该节点所经分支上的所有节点；
子孙：以某节点为根的子树中任一节点都称为该节点的子孙。
森林：由m（m>=0）棵互不相交的树的集合称为森林；

无序树：树中任意节点的子节点之间没有顺序关系，这种树称为无序树，也称为自由树；
有序树：树中任意节点的子节点之间有顺序关系，这种树称为有序树；
二叉树：每个节点最多含有两个子树的树称为二叉树；
完全二叉树：对于一颗二叉树，假设其深度为d(d>1)。除了第d层外，其它各层的节点数目均已达最大值，且第d层所有节点从左向右连续地紧密排列，这样的二叉树被称为完全二叉树，其中满二叉树的定义是所有叶节点都在最底层的完全二叉树;
平衡二叉树（AVL树）：当且仅当任何节点的两棵子树的高度差不大于1的二叉树；
排序二叉树（二叉查找树（英语：Binary Search Tree），也称二叉搜索树、有序二叉树）；
霍夫曼树（用于信息编码）：带权路径最短的二叉树称为哈夫曼树或最优二叉树；
B树：一种对读写操作进行优化的自平衡的二叉查找树，能够保持数据有序，拥有多余两个子树。

--- 树的存储 ---
顺序存储：将数据结构存储在固定的数组中，然在遍历速度上有一定的优势，但因所占空间比较大，是非主流二叉树。二叉树通常以链式存储。
链式存储： 由于对节点的个数无法掌握，常见树的存储表示都转换成二叉树进行处理，子节点个数最多为2


二叉树的性质(特性)
性质1: 在二叉树的第i层上至多有2^(i-1)个结点（i>0）
性质2: 深度为k的二叉树至多有2^k - 1个结点（k>0）
性质3: 对于任意一棵二叉树，如果其叶结点数为N0，而度数为2的结点总数为N2，则N0=N2+1;
性质4:具有n个结点的完全二叉树的深度必为 log2(n+1)
性质5:对完全二叉树，若从上至下、从左至右编号，则编号为i 的结点，其左孩子编号必为2i，其右孩子编号必为2i＋1；其双亲的编号必为i/2（i＝1 时为根,除外）

(1)完全二叉树——若设二叉树的高度为h，除第 h 层外，其它各层 (1～h-1) 的结点数都达到最大个数，第h层有叶子结点，并且叶子结点都是从左到右依次排布，这就是完全二叉树。
(2)满二叉树——除了叶结点外每一个结点都有左右子叶且叶子结点都处在最底层的二叉树。
"""


class Node:

    def __init__(self, element=-1, lchild=None, rchild=None):
        self.element = element
        self.lchild = lchild
        self.rchild = rchild


class Tree:
    """
    --- 二叉树的遍历 ---
树的两种重要的遍历模式是深度优先遍历和广度优先遍历,深度优先一般用递归，广度优先一般用队列。一般情况下能用递归实现的算法大部分也能用堆栈来实现。

    --- 深度优先遍历 ---
    对于一颗二叉树，深度优先搜索(Depth First Search)是沿着树的深度遍历树的节点，尽可能深的搜索树的分支。
深度遍历有重要的三种方法: 先序遍历（preorder），中序遍历（inorder）和后序遍历（postorder）


    """
    def __init__(self, root=None):
        self.root = root

    def add(self, element):
        node = Node(element)
        if self.root is None:
            self.root = node
        else:
            queue = []
            queue.append(self.root)
            while queue:
                # 弹出队列的第一个元素
                cursor = queue.pop(0)
                if cursor.lchild is None:
                    cursor.lchild = node
                    return

                elif cursor.rchild is None:
                    cursor.rchild = node
                    return

                else:
                    # 如果左右子树都不为空，加入队列继续判断
                    queue.append(cursor.lchild)
                    queue.append(cursor.rchild)

    def preorder(self, root):
        """
        先序遍历 在先序遍历中，我们先访问根节点，然后递归使用先序遍历访问左子树，再递归使用先序遍历访问右子树
        根节点->左子树->右子树
        """
        if root is None:
            return
        print(root.element)
        self.preorder(root.lchild)
        self.preorder(root.rchild)

    def inorder(self, root):
        """
        中序遍历 在中序遍历中，我们递归使用中序遍历访问左子树，然后访问根节点，最后再递归使用中序遍历访问右子树
        左子树->根节点->右子树
        """
        if root is None:
            return
        self.inorder(root.lchild)
        print(root.element)
        self.inorder(root.rchild)

    def postorder(self, root):
        """
        后序遍历 在后序遍历中，我们先递归使用后序遍历访问左子树和右子树，最后访问根节点
        左子树->右子树->根节点
        """
        if root is None:
            return
        self.preorder(root.lchild)
        self.preorder(root.rchild)
        print(root.element)

    def breadth_travel(self, root):
        if root is None:
            return
        queue = []
        queue.append(root)
        while queue:
            node = queue.pop(0)
            print(node.element)
            if node.lchild is not None:
                queue.append(node.lchild)
            if node.rchild is not None:
                queue.append(node.rchild)


if __name__ == '__main__':
    pass
