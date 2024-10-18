from torch.utils.data import Dataset
import numpy as np
import math


class DomainSupervisedDataset(Dataset):
    def __init__(self, path, n = None, batchsize = None, t_max = None):
        self.path = path
        self.t_max = t_max
        self.data = self.__exact(self.t_max)
        if n == None or n>len(self.data):
            self.n = len(self.data)
        else:
            self.n = n
            self.data = self.data[np.random.choice(self.data.shape[0], n, replace=False), :]
        if batchsize != None and batchsize > n:
            batchsize = n
        self.batchsize = batchsize
        
        
    def __len__(self):
        return 1 if self.batchsize == None else int(math.ceil(self.n/self.batchsize))

    def __getitem__(self, index):
        if self.batchsize == None:
            return self.data
        else:
            start = index*self.batchsize
            end = (index+1)*self.batchsize
            if end >= self.n:
                end = self.n - 1
            return self.data[start:end]
        
    def __str__(self):
        s = f"DomainSupervisedDataset(n={self.n}, batchsize={self.batchsize}, t_max={self.t_max})"
        return s
        
    def __exact(self, t_max = None):
        sol = []
        with open(self.path, "r") as f:
            for line in f:
                line = line.split(",")
                s = line[3].strip()  # Assuming the solution is now in the 4th column
                s = s.replace('"', '').replace("{", "").replace("}", "").replace("*^", "E")
                s = float(s)
                x = float(line[0])
                y = float(line[1])  # Add y coordinate
                t = float(line[2])  # t is now in the 3rd column
                if t_max != None:
                    if t <= t_max:
                        sol.append([x, y, t, s])  # Include y in the solution
                else:
                    sol.append([x, y, t, s])  # Include y in the solution
        return np.array(sol)

class DomainDataset(Dataset):
    def __init__(self, xmin, xmax, n, batchsize = None, shuffle = True, period = 1, seed = 1234):
        self.xmin = np.array(xmin, dtype="f")
        self.xmax = np.array(xmax, dtype="f")
        self.dim = len(xmin)
        self.n = n
        self.shuffle = shuffle
        self.period = period
        if batchsize != None:
            self.n_points_per_axis = batchsize
        else:
            self.n_points_per_axis = n
        self.rng = np.random.default_rng(seed)
        self.seed = self.rng.integers(0, 10000)
        self.rng2 = np.random.default_rng(self.seed)
        if shuffle:
            self.period_counter = 0 
        self.counter = 0

    def __len__(self):
        return 1 if self.n_points_per_axis == self.n else int(math.ceil(self.n/self.n_points_per_axis))
    
    def _sample_items(self, length):
        x = self.rng2.uniform(low=self.xmin[0], high=np.nextafter(self.xmax[0], self.xmax[0]+1), size=(length, 1))
        for i in range(1, self.dim):
            s = self.rng2.uniform(low=self.xmin[i], high=np.nextafter(self.xmax[i], self.xmax[i]+1), size=(length, 1))
            if i == self.dim - 1:
                #sort for ascending time
                s.sort()
            x = np.hstack((x, s))
        return x
    
    def __getitem__(self, idx):
        if self.counter >= self.n:
            #time to reset the dataset generator
            self.rng2 = np.random.default_rng(self.seed)
            self.counter = 0
            if self.shuffle:
                self.period_counter += 1
        if self.shuffle and self.period_counter >= self.period:
            #time to change the dataset generator
            self.seed = self.rng.integers(0, 10000)
            self.rng2 = np.random.default_rng(self.seed)
            self.counter = 0
            self.period_counter = 0
        length = self.n - self.counter if self.n - self.counter < self.n_points_per_axis else self.n_points_per_axis
        x = self._sample_items(length)
        self.counter += length 
        return x
    
    def __str__(self):
        s = f"DomainDataset({self.xmin}, {self.xmax}, n={self.n}, shuffle={self.shuffle}, period={self.period})"
        return s
    

class ICDataset(DomainDataset):
    def __init__(self, xmin, xmax, n, batchsize = None, shuffle = True, period = 1):
        super().__init__(xmin, xmax, n, batchsize = batchsize, shuffle = shuffle, period = period)
        
    def _sample_items(self, length):

        x = self.rng2.uniform(low=self.xmin[0], high=np.nextafter(self.xmax[0], self.xmax[0]+1), size=(length, ))
        for i in range(1, self.dim):
            s = self.rng2.uniform(low=self.xmin[i], high=np.nextafter(self.xmax[i], self.xmax[i]+1), size=(length, ))
            x = np.vstack((x, s))
        x = np.vstack((x, np.zeros(length, )))
        return x.T
    
    
    def __str__(self):
        s = f"ICDataset({self.xmin}, {self.xmax}, n={self.n}, shuffle={self.shuffle}, period={self.period})"
        return s


class BCDataset(DomainDataset):
    def __init__(self, xmin, xmax, n, rand=True, shuffle=False, period=1):
        super().__init__(xmin, xmax, n)
    
    #TO BE IMPLEMENTED