import numpy as np

class NaiveBayesModel():
    def __init__(self,alpha=1):
        self.pre_prob = []
        self.condition_prob = []
        self.classes = None
        self.alpha = alpha

    def learning(self,x:np.ndarray,y:np.ndarray):
        # x [N.D]
        # y [N]
        self.classes = np.unique(y)
        for i in self.classes:
            y_c = (y == i)
            self.pre_prob.append((np.sum(y_c)+self.alpha)/(len(y)+self.alpha*len(self.classes)))
           
            x_features = [np.unique(x[:,a]) for a in range(x.shape[1])]
            selected_x = x[y_c,:]
            cache = []
            for j in range(len(x_features)):
                dic = {}
                for k in range(len(x_features[j])):
                    dic[x_features[j][k]] = (np.sum(selected_x[:,j]==x_features[j][k])+self.alpha)/(len(selected_x)+self.alpha*len(x_features[j]))
                cache.append(dic)
            self.condition_prob.append(cache)


                
if __name__ == "__main__":
    pass