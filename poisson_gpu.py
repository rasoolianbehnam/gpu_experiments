
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
from scipy.sparse import csr_matrix, lil_matrix, coo_matrix, eye
import time
import os.path
def mat_pow(a, k, b):
    if k == 0:
        m = np.eye(a.shape[0])
        b = m * 1.
    elif k % 2:
        m, b = mat_pow(a, k-1, b)
        m = a.dot(m)
        b += a
    else:
        m, b = mat_pow(a, k // 2, b)
        m = m.dot(m)
        b += a
    print(k)
    return m, b

def convert_sparse_matrix_to_sparse_tensor(coo):
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)

def save_sparse_csr(filename, array):
        np.savez(filename, data=array.data, \
                indices=array.indices, indptr=array.indptr, shape=array.shape)

def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])

class poisson_vectorized:
    def __init__(self, n1, n2, n3, w, num_iterations=40, h=1e3, method='ndarray',\
            upper_lim=3):
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.imax = n1 - upper_lim
        self.jmax = n2 - upper_lim
        self.kmax = n3 - upper_lim

        self.method = method
        self.kernels = {}
        #keys = ['poisson', 'x_gradient', 'y_gradient', 'average_x']
        keys = ['poisson']
        if self.method == 'lil' or method == 'coo1':
            for key in keys:
                self.kernels[key] = lil_matrix((n1*n2*n3, n1*n2*n3), dtype='float32')
        elif method == 'coo':
            for key in keys:
                self.kernels[key] = np.zeros((n1*n2*n3, n1*n2*n3), dtype='float32')
        elif self.method == 'ndarray':
            for key in keys:
                self.kernels[key] = np.zeros((n1*n2*n3, n1*n2*n3), dtype='float32')
        print("method used is %s"%(self.method))
        self.w = w
        self.h = h
        self.h2 = h**2
        self.num_iterations = num_iterations
        print("Starting to create kernels...")
        for I in range(0, n1*n2*n3):
            k = I % n3
            s1 = (I - k) // n3
            j = s1 % n2
            i = (s1 - j) // n2
            #print(I, i, j, k)
            if (i >= 1 and i < self.imax-1 and 
                j >= 1 and j < self.jmax-1 and 
                k >= 1 and k < self.kmax-1):
                #print(I, i, j, k)
                if (I % 1000 == 0):
                    print("%d / %d"%(I, n1*n2*n3))
                self.kernels['poisson'][I, :] =self.w*1./6* \
                      ( self.kernels['poisson'][I-1, :] \
                      + self.kernels['poisson'][I-n3, :] \
                      + self.kernels['poisson'][I-n3*n2, :])
                self.kernels['poisson'][I, I+1] += self.w*1./6
                self.kernels['poisson'][I, I+n3] += self.w*1./6
                self.kernels['poisson'][I, I+n2*n3] += self.w*1./6
                self.kernels['poisson'][I, I] += 1 - self.w

#                self.kernels['x_gradient'][I, I] = -1
#                self.kernels['x_gradient'][I, I+n2*n3] = 1
#
#                self.kernels['y_gradient'][I, I] = -1
#                self.kernels['y_gradient'][I, I + n3] = 1
#            elif (i >= 2 and i < self.imax+1 and 
#                  j >= 1 and j < self.jmax and 
#                  k >= 1 and k < self.kmax-1):
#                self.kernels['average_x'][I, I] = 1
#                self.kernels['average_x'][I, I+n3] = 1
#                self.kernels['average_x'][I, I-n2*n3] = 1
#                self.kernels['average_x'][I, I+n3-n2*n3] = 1
            

        print("Finished creating kernels.")
        if self.method == 'coo':
            print("Starting coo conversion")
            for key in self.kernels:
                self.kernels[key] = coo_matrix(self.kernels[key], (n1*n2*n3, n1*n2*n3), dtype='float32')
        # uncomment in order to make fast1 work
        # fast1 is less efficient that fast due to chain matrix multiplicaiton order
        # rule.
        print("calculating matrix powers")

        if self.method == 'coo':
            self.B = lil_matrix(eye(n1*n2*n3, n1*n2*n3), dtype='float32')
        elif self.method == 'ndarray':
            self.B = np.eye(n1*n2*n3, n1*n2*n3)

        A_file_name = 'self_A_%d_%d_%d_%d_coo.npz'%(n1, n2, n3, self.num_iterations)
        B_file_name = 'self_B_%d_%d_%d_%d_coo.npz'%(n1, n2, n3, self.num_iterations)
        A_loaded = False
        if os.path.isfile(A_file_name):
            print("A file exitst.")
            self.A = load_sparse_csr(A_file_name)
            A_loaded = True
        B_loaded = False
        if os.path.isfile(B_file_name):
            print("B file exitst.")
            self.B = load_sparse_csr(B_file_name)
            B_loaded = True
        if A_loaded and B_loaded:
            print("A and B has been found!!")
            if self.method == 'ndarray':
                self.A = self.A.todense()
                self.B = self.B.todense()
        else:
            self.A = self.kernels['poisson']
            for kk in range(self.num_iterations-1):
                print(kk)
                self.B += self.A
                self.A = self.kernels['poisson'].dot(self.A)
            print('converting large A and B to coo format to store')
            self.B = coo_matrix(self.B, (n1*n2*n3, n1*n2*n3), dtype='float32')
            self.A = coo_matrix(self.B, (n1*n2*n3, n1*n2*n3), dtype='float32')

            print("saving self.B: ")
            save_sparse_csr(B_file_name, self.B.tocsr())
            print("saving self.A: ")
            save_sparse_csr(A_file_name, self.A.tocsr())
            print("finished!!")


            
    def poisson_fast(self, V, g):
        out = V
        g = self.w * self.h2 * g / 6
        for I in range(0, self.n1*self.n2*self.n3):
            k = I % self.n3
            s1 = (I - k) // self.n3
            j = s1 % self.n2
            i = (s1 - j) // self.n2
            #print(I, i, j, k)
            if (i >= 1 and i < self.imax-1 and 
                j >= 1 and j < self.jmax-1 and 
                k >= 1 and k < self.kmax-1):
                #print(I, i, j, k)
                g[I] +=self.w*1./6* \
                      ( g[I-1]
                      + g[I-self.n3]
                      + g[I-self.n3*self.n2])
        for kk in range(self.num_iterations):
            out = self.kernels['poisson'].dot(out) - g
        return out

    def poisson_fast_gpu(self, V, g, A, sess):
        out = V
        g = self.w * self.h2 * g / 6
        for kk in range(self.num_iterations):
            out = tf.sparse_tensor_dense_matmul(A, out) - g
            #out = tf.matmul(A, out) - g
        sess.run(out)
        return sess, out

    #can be numerically unstable
    def poisson_fast1(self, V, g):
        g = self.w * self.h2 * g / 6.
        for I in range(0, self.n1*self.n2*self.n3):
            k = I % self.n3
            s1 = (I - k) // self.n3
            j = s1 % self.n2
            i = (s1 - j) // self.n2
            #print(I, i, j, k)
            if (i >= 1 and i < self.imax-1 and 
                j >= 1 and j < self.jmax-1 and 
                k >= 1 and k < self.kmax-1):
                #print(I, i, j, k)
                g[I] +=self.w*1./6* \
                      ( g[I-1]
                      + g[I-self.n3]
                      + g[I-self.n3*self.n2])
        return self.A.dot((V)) \
                - self.B.dot(g) \

    #@jit
    def poisson_brute2(self, V, g):
        for kk in range(self.num_iterations):
            temp = V * 1.
            for I in range(0, self.n1*self.n2*self.n3):
                k = I % self.n3
                s1 = (I - k) // self.n3
                j = s1 % self.n2
                i = (s1 - j) // self.n2
                if (i*j*k==0 or k >= self.kmax - 1 or j >= self.jmax - 1 or i >= self.imax - 1):
                    V[k + self.n3 * (j + self.n2 * i)] = 0 
                    continue
                r =  temp[I-1] / 6.+   temp[I+1] / 6.+  temp[I+self.n3] / 6.+   temp[I-self.n3] / 6.+   temp[I+self.n2*self.n3] / 6. + temp[I-self.n2*self.n3] / 6.  - temp[I] - self.h2 * g[I] / 6.
                r = self.w * r
                V[I] += r
        return V

    #@jit
    def poisson_brute4(self, V, g):
        for kk in range(self.num_iterations):
            temp = V
            for I in range(0, self.n1*self.n2*self.n3):
                k = I % self.n3
                s1 = (I - k) // self.n3
                j = s1 % self.n2
                i = (s1 - j) // self.n2
                if (i*j*k==0 or k >= self.kmax - 1 or j >= self.jmax - 1 or i >= self.imax - 1):
                    V[k + self.n3 * (j + self.n2 * i)] = 0 
                    continue
                r =  temp[I-1] / 6.+   temp[I+1] / 6.+  temp[I+self.n3] / 6.+   temp[I-self.n3] / 6.+   temp[I+self.n2*self.n3] / 6. + temp[I-self.n2*self.n3] / 6.  - temp[I] - self.h2 * g[I] / 6.
                r = self.w * r
                V[I] += r
        return V
    
    #@jit
    def poisson_brute(self, V, g):
        for kk in range(self.num_iterations):
            temp = V * 1.
            for i in range(1, self.imax-1):
                for j in range(1, self.jmax-1):
                    for k in range(1, self.kmax-1):
                        r = temp[i+1, j, k] / 6. + temp[i-1, j, k] / 6. + temp[i, j+1, k] / 6. + temp[i, j-1, k] / 6. + temp[i ,j, k+1] / 6. + temp[i, j, k-1] / 6. - temp[i, j, k] - self.h2 * g[i, j, k] / 6.
                        r = self.w * r
                        V[i, j, k] += r
        return V

    def poisson_brute3(self, temp, g):
        for kk in range(self.num_iterations):
            r = self.w * (( temp[2:self.imax,   1:self.jmax-1, 1:self.kmax-1] \
                + temp[0:self.imax-2, 1:self.jmax-1, 1:self.kmax-1] \
                + temp[1:self.imax-1, 2:self.jmax,   1:self.kmax-1] \
                + temp[1:self.imax-1, 0:self.jmax-2, 1:self.kmax-1] \
                + temp[1:self.imax-1, 1:self.jmax-1, 2:self.kmax  ] \
                + temp[1:self.imax-1, 1:self.jmax-1, 0:self.kmax-2])) / 6. \
                + (1-self.w) * temp[1:self.imax-1, 1:self.jmax-1, 1:self.kmax-1]  \
                - self.w * self.h2 * g[1:self.imax-1, 1:self.jmax-1, 1:self.kmax-1] / 6.
            temp[1:self.imax-1, 1:self.jmax-1, 1:self.kmax-1] = r
        return temp

    def poisson_brute3_gpu(self, V, g, sess):
        out = tf.Variable(tf.float32, V)
        #for kk in range(self.num_iterations):
        #    out[1:self.imax-1, 1:self.jmax-1, 1:self.kmax-1].assign(out[1:self.imax-1, 1:self.jmax-1, 1:self.kmax-1] + 1)
        sess.run(out)
        return sess, out

    def apply_kernel(self, V, key):
        return self.kernels[key].dot(V)
    
    def electric_field_elements(self, V):
        out = np.zeros_like(V)
        out[1:self.imax+1, :, :] = (V[1:self.imax+1, :, :] - V[2:self.imax+2, :, :]) / self.h
        return out

    def average_x(self, V, out):
        for j in range(1, self.jmax):
            out[1, j, :] = V[1, j, :] + V[1, j+1, :] + V[self.imax, j, :] + \
                    V[self.imax, j+1, :]

    def average_x_fast(self, V, out):
        out[1, 1:self.jmax, :] = V[1, 1:self.jmax] + V[1, 2:self.jmax+1, :] \
                + V[self.imax, 1:self.jmax, :] + V[self.imax, 2:self.jmax+1, :]


def rel_error(x, y):
  """ re-turns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

def stat_diff(V1, V2, n1, n2, n3):
    print("relative error",  rel_error(V1, V2))
    #print(np.where(np.abs(V1 - V2) > 20))

def main():
    imax = 20
    jmax = 16
    kmax = 8
    upper_lim = 1
    n1 = imax + upper_lim
    n2 = jmax + upper_lim
    n3 = kmax + upper_lim
    w = 1.843
    pv = poisson_vectorized(n1, n2, n3, w=w, num_iterations=40, method='coo'\
                                    , upper_lim=upper_lim)
    V1 = (np.random.rand(n1 * n2 * n3) * 30  + 1).astype('float32')
    g = (np.random.rand(n1 * n2 * n3) * 2e5  + -1e5).astype('float32')

    for I in range(0, n1*n2*n3):
        k = I % n3
        s1 = (I - k) // n3
        j = s1 % n2
        i = (s1 - j) // n2
        #print(I, i, j, k)
        if (i*j*k==0 or k >= kmax - 1 or j >= jmax - 1 or i >= imax - 1):
            g[I] = 0
            V1[I] = 0
    V1_reshaped = V1.reshape(n1, n2, n3)
    g_reshaped = g.reshape(n1, n2, n3)

                   
    print("Starting poisson fast")
    start = time.time()
    a = pv.poisson_fast(V1 * 1., g)
    print("Time taken: %f"%(time.time() - start))

    print("Starting poisson brute 2")
    start = time.time()
    b = pv.poisson_brute(V1.reshape(n1, n2, n3) , g.reshape(n1, n2, n3))
    print("Time taken: %f"%(time.time() - start))

    stat_diff(a, b.reshape(n1*n2*n3), n1, n2, n3)

if __name__ == "__main__":
    main()
