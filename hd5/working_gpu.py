
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
from scipy.sparse import csr_matrix, lil_matrix, coo_matrix, eye
import time as Time
import os.path
import h5py

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

def save_matrix_as_txt(mat, file_name):
    print("writing to txt file %s"%(file_name))
    x, y = mat.nonzero()
    nonzero = zip(x, y)
    with open(file_name, 'w+') as myA:
        for r, k in nonzero:
            myA.write("%d %d %8.4f\n"%(r, k, mat[r, k]))

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
    def __init__(self, n1, n2, n3, w, num_iterations=40, h=1e-3, method='ndarray',\
            upper_lim=3, want_cuda=False):
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.imax = n1 - upper_lim
        self.jmax = n2 - upper_lim
        self.kmax = n3 - upper_lim

        self.method = method
        self.kernels = {}
        #keys = ['poisson', 'x_gradient', 'y_gradient', 'average_x']
        keys = ['poisson', 'g']
        if self.method == 'lil' or method == 'coo1':
            for key in keys:
                self.kernels[key] = lil_matrix((n1*n2*n3, n1*n2*n3), dtype='float64')
        elif method == 'coo':
            for key in keys:
                self.kernels[key] = np.zeros((n1*n2*n3, n1*n2*n3), dtype='float64')
        elif self.method == 'ndarray':
            for key in keys:
                self.kernels[key] = np.zeros((n1*n2*n3, n1*n2*n3), dtype='float64')
        print("method used is %s"%(self.method))
        self.w = w
        self.h = h
        self.h2 = h**2
        self.num_iterations = num_iterations
        print("Starting to create kernels...")
        for I in range(0, n1*n2*n3):
            if (I % 1000 == 0):
                print("%d / %d"%(I, n1*n2*n3))
            k = I % n3
            s1 = (I - k) // n3
            j = s1 % n2
            i = (s1 - j) // n2
            #print(I, i, j, k)
            if (i >= 1 and i < self.imax-1 and 
                j >= 1 and j < self.jmax-1 and 
                k >= 1 and k < self.kmax-1):
                #print(I, i, j, k)
                self.kernels['poisson'][I, :] =self.w*1./6* \
                      ( self.kernels['poisson'][I-1, :] \
                      + self.kernels['poisson'][I-n3, :] \
                      + self.kernels['poisson'][I-n3*n2, :])
                self.kernels['poisson'][I, I+1] += self.w*1./6
                self.kernels['poisson'][I, I+n3] += self.w*1./6
                self.kernels['poisson'][I, I+n2*n3] += self.w*1./6
                self.kernels['poisson'][I, I] += 1 - self.w

                self.kernels['g'][I, :] = self.w * 1./6*\
                        ( self.kernels['g'][I-1,:] \
                        + self.kernels['g'][I-self.n3, :] \
                        + self.kernels['g'][I-self.n3*self.n2, :] \
                        )
                self.kernels['g'][I, I] += 1


            else:
                self.kernels['poisson'][I, I] = 1
            

        print("w = %f"%(self.w));
        print("sum of kernel: %f"%np.sum(self.kernels['poisson']))
        print("Finished creating kernels.")
        if self.method == 'coo':
            print("Starting coo conversion")
            for key in self.kernels:
                self.kernels[key] = coo_matrix(self.kernels[key], (n1*n2*n3, n1*n2*n3), dtype='float64')



        print("calculating matrix powers")
        file_name = 'data_%d_%d_%d_%d.txt'%(n1, n2, n3, self.num_iterations)
        A_file = 'A_%d_%d_%d_%d.txt'%(n1, n2, n3, self.num_iterations)
        B_file = 'B_%d_%d_%d_%d.txt'%(n1, n2, n3, self.num_iterations)
        #if os.path.isfile(file_name):
        #    print("%s exists"%file_name)
        #    h5f = h5py.File(file_name, 'r')
        #    self.A = h5f['A'][:]
        #    self.B = h5f['B'][:]
        #else:

        start = Time.time()

        I_t = tf.placeholder(tf.float64, [n1*n2*n3, n1*n2*n3])
        B_t = tf.eye(n1*n2*n3, dtype=tf.float64)
        B_t = B_t + I_t
        A_t = tf.matmul(I_t, I_t)
        for kk in range(self.num_iterations-2):
            B_t = B_t + A_t
            A_t = tf.matmul(I_t, A_t)

        variables = [A_t, B_t]
        feed_dict = {I_t: self.kernels['poisson']}
        with tf.Session() as sess:
            with tf.device('/cpu:0'):
                self.A, self.B = sess.run(variables, feed_dict=feed_dict)

        time_taken2 = Time.time() - start
        print("time taken to calc powers:", time_taken2)
        #start = Time.time()
        #self.A = self.kernels['poisson']
        #self.B = np.eye(n1*n2*n3, n1*n2*n3)
        #for kk in range(self.num_iterations-1):
        #    if (kk % 10 == 0):
        #        print(kk)
        #    self.B += self.A
        #    self.A = self.kernels['poisson'].dot(self.A)
        #time_taken2 = Time.time() - start
        #print("diff of AA and A", rel_error(self.A, self.AA))
        #print("diff of AA and A", rel_error(self.B, self.BB))
        #np.save(A_file_name, self.A)
        #np.save(B_file_name, self.B)
        #h5f = h5py.File(file_name, 'w')
        #h5f.create_dataset('A', data=self.A)
        #h5f.create_dataset('B', data=self.B)
        #np.save("mardas", self.A)
        #h5f.close()
        print("time taken to calc powers:", time_taken2)
        save_matrix_as_txt(self.A, A_file)
        save_matrix_as_txt(self.B, B_file)

        print("sum of A: ", np.sum(self.A))
        print("sum of B: ", np.sum(self.B))
        print("finished!!")


            
    def poisson_fast_one_loop(self, V, g):
        out = V
        g = self.w * self.h2 * g / 6
        for I in range(0, self.n1*self.n2*self.n3):
            k = I % self.n3
            s1 = (I - k) // self.n3
            j = s1 % self.n2
            i = (s1 - j) // self.n2
            #print(I, i, j, k)
            if (i >= 1 and i < self.imax+1 and 
                j >= 1 and j < self.jmax+1 and 
                k >= 1 and k < self.kmax-1):
                #print(I, i, j, k)
                g[I] +=self.w*1./6* \
                      ( g[I-1]
                      + g[I-self.n3]
                      + g[I-self.n3*self.n2])
            else:
                g[I] = 0
        for kk in range(self.num_iterations):
            out = self.kernels['poisson'].dot(out) - g
        return out

    def poisson_fast_one_loop_gpu(self, V, g, A, sess):
        out = V
        g = self.w * self.h2 * g / 6
        for kk in range(self.num_iterations):
            out = tf.sparse_tensor_dense_matmul(A, out) - g
            #out = tf.matmul(A, out) - g
        sess.run(out)
        return sess, out

    #can be numerically unstable
    def poisson_fast_no_loop_old(self, V, g):
        g = self.w * self.h2 * g / 6.
        for I in range(0, self.n1*self.n2*self.n3):
            k = I % self.n3
            s1 = (I - k) // self.n3
            j = s1 % self.n2
            i = (s1 - j) // self.n2
            #print(I, i, j, k)
            if (i >= 1 and i < self.imax+1 and 
                j >= 1 and j < self.jmax+1 and 
                k >= 1 and k < self.kmax-1):
                #print(I, i, j, k)
                g[I] +=self.w*1./6* \
                      ( g[I-1]
                      + g[I-self.n3]
                      + g[I-self.n3*self.n2])
            else:
                g[I] = 0
        return self.A.dot((V)) \
                - self.B.dot(g) \

    #can be numerically unstable
    def poisson_fast_no_loop(self, V, g):
        g = self.w * self.h2 * g / 6.
        g = self.kernels['g'].dot(g)
        return self.A.dot((V)) \
                - self.B.dot(g) \

    def poisson_fast_no_loop_torch(self, V, g):
        g = g * self.w * self.h2 / 6.
        g = torch.mm(self.kernels['g_torch'], g)
        return torch.mm(self.Atorch, V) - torch.mm(self.Btorch, g)

    def poisson_fast_no_loop_gpu(self, V, g):
        g = self.w * self.h2 * g / 6.
        for I in range(0, self.n1*self.n2*self.n3):
            k = I % self.n3
            s1 = (I - k) // self.n3
            j = s1 % self.n2
            i = (s1 - j) // self.n2
            #print(I, i, j, k)
            if (i >= 1 and i < self.imax+1 and 
                j >= 1 and j < self.jmax+1 and 
                k >= 1 and k < self.kmax-1):
                #print(I, i, j, k)
                g[I] +=self.w*1./6* \
                      ( g[I-1]
                      + g[I-self.n3]
                      + g[I-self.n3*self.n2])
        Atf = tf.placeholder(tf.float64, shape=[self.n1*self.n2*self.n3, self.n1*self.n2*self.n3])
        Btf = tf.placeholder(tf.float64, shape=[self.n1*self.n2*self.n3, self.n1*self.n2*self.n3])
        Vtf = tf.placeholder(tf.float64, shape=[self.n1*self.n2*self.n3, 1])
        gtf = tf.placeholder(tf.float64, shape=[self.n1*self.n2*self.n3, 1])
        out = tf.matmul(Atf, Vtf) - tf.matmul(Btf, gtf)
        with tf.Session() as sess:
            out = sess.run(out, feed_dict = {Atf:self.A, Btf:self.B, Vtf:V, gtf:g})
        return out

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
    def poisson_brute_main_flat(self, V, g):
        for kk in range(self.num_iterations):
            temp = V
            for I in range(0, self.n1*self.n2*self.n3):
                k = I % self.n3
                s1 = (I - k) // self.n3
                j = s1 % self.n2
                i = (s1 - j) // self.n2
                if (i*j*k==0 or k >= self.kmax - 1 or j >= self.jmax + 1 or i >= self.imax + 1):
                    continue
                r =  temp[I-1] / 6.+   temp[I+1] / 6.+  temp[I+self.n3] / 6.+   temp[I-self.n3] / 6.+   temp[I+self.n2*self.n3] / 6. + temp[I-self.n2*self.n3] / 6.  - temp[I] - self.h2 * g[I] / 6.
                r = self.w * r
                V[I] += r
        return V
    
    #@jit
    def poisson_brute_main(self, V, g):
        for kk in range(self.num_iterations):
            temp = V
            for i in range(1, self.imax+1):
                for j in range(1, self.jmax+1):
                    for k in range(1, self.kmax-1):
                        r = temp[i+1, j, k] / 6. + temp[i-1, j, k] / 6. + temp[i, j+1, k] / 6. + temp[i, j-1, k] / 6. + temp[i ,j, k+1] / 6. + temp[i, j, k-1] / 6. - temp[i, j, k] - self.h2 * g[i, j, k] / 6.
                        r = self.w * r
                        V[i, j, k] += r
        return V

    def poisson_brute_vectorized(self, temp, g):
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
        out = tf.Variable(tf.float64, V)
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

def max_rel_error_loc(x, y):
  """ re-turns relative error """
  a = np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y)))
  return a

def stat_diff(V1, V2, text=""):
    print("%-30s relative error"%text,  rel_error(V1, V2))
    #print(np.where(np.abs(V1 - V2) > 20))

    
if __name__ == "__main__":
    main()
