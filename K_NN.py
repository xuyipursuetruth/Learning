import numpy as np

# 最原始形式
class k_NN_v1():
    def __init__(self,k=1):
        self.k = k

    def forward(self,train_x,train_y,x):
        length = []
        id = []

        for i in range(len(train_x)):
            l = np.sum(np.abs(train_x[i]-x)) # 曼哈顿距离(可修改距离度量)
            if (len(length) >= self.k):
                if (l < max(length)):
                    index = length.index(max(length))
                    length[index] = l
                    id[index] = i
            else:
                length.append(l)
                id.append(i)

        c = []
        for i in range(len(id)):
            c.append(train_y[id[i]])
        cls = list(set(c))
        num = c.count(cls[0])
        y = cls[0]
        for i in range(1,len(cls)):
            if (num < c.count(cls[i])):
                num = c.count(cls[i])
                y = cls[i]
        return y

# 使用KD树
class kdNode():
    def __init__(self,alldata,mid,axis):
        self.data = alldata[mid]
        self.left_tree = None
        self.right_tree = None
        self.axis = axis

class kdTree():
    def __init__(self,train_data:np.ndarray):
        self.train_data = train_data # [N,D]
        self.n_data = train_data.shape[0]
        self.dim = train_data.shape[1]
        self.axis = 0
        self.root = self.create_tree(self.train_data)

    def update_axis(self):
        self.axis += 1
        if self.axis >= self.dim:
            self.axis = 0

    def create_tree(self,train_data:np.ndarray):
        # train_data [N,D]        
        n_data = train_data.shape[0]

        if n_data == 0:
            return None

        if n_data == 1:
            return kdNode(train_data,0,self.axis)
        
        mid = n_data // 2
        a = train_data[np.argpartition(train_data[:,self.axis],mid)]
        node = kdNode(a,mid,self.axis)
        self.update_axis()
        node.left_tree = self.create_tree(a[:mid])
        node.right_tree = self.create_tree(a[mid+1:])
        
        return node

    def search_nearest(self,x,root:kdNode):
        # x [D]
        path = []
        node = root
        while node != None:
            path.append(node)
            if node.data[node.axis] > x[node.axis]:
                node = node.left_tree
            else:
                node = node.right_tree

        path.reverse()
        node = path[0]
        d0 = np.linalg.norm(node.data-x)
        nearest_node = node.data

        for i in range(1,len(path)):
            node = path[i]
            d1 = np.linalg.norm(node.data-x)
            if d1 < d0:
                nearest_node = node.data
                d0 = d1

            d2 = x[node.axis]-node.data[node.axis]

            if abs(d2) < d0:
                if d2 >= 0:
                    if node.left_tree != None:
                        nearest_node1,d3 = self.search_nearest(x,node.left_tree)
                        if d3 < d0:
                            nearest_node = nearest_node1
                            d0 = d3
                else:
                    if node.right_tree != None:
                        nearest_node1,d3 = self.search_nearest(x,node.right_tree)
                        if d3 < d0:
                            nearest_node = nearest_node1
                            d0 = d3
        return nearest_node,d0
    
if __name__ == "__main__":
    eg = kdTree(np.array([[2,3],[5,4],[9,6],[4,7],[8,1],[7,2]]))
    print(eg.search_nearest(np.array([6,6]),eg.root))