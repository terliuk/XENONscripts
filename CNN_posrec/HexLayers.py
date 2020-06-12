import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

class ConvHex(tf.keras.layers.Layer):
    def __init__(self, num_outputs, kernel_radius, activation="elu", ignore_out_coords = [], **kwargs):
        # ignore out = list of tuples that will be excluded for outputs
        # XENONnT is not exactly hexagonal and some pixels in corners are missing
        super(ConvHex, self).__init__(**kwargs)
        self.num_outputs = num_outputs
        self.kernel_radius = kernel_radius
        self.activation = activation
        self.winitializer = tf.keras.initializers.GlorotNormal()
        self.ignore_out = ignore_out_coords
    
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'num_outputs': self.num_outputs,
            'kernel_radius': self.kernel_radius,
            'activation': self.activation,
            'ignore_out_coords': self.ignore_out
        })
        return config
    
    def MakeHexMaskIndices(self,radius):
        n_ind = int(1 + 6*radius*(1+radius)/2)
        moves = np.array([[ 1, 1],[ 2, 0],[ 1,-1],
                          [-1,-1],[-2, 0],[-1, 1]])
        one_out = []
        one_out.append([2*radius, radius]) # central "pixel"
        for il in range(1, radius+1):
            # layers of HEX pattern
            s_coord = np.array([[2*radius - 2*il, radius]])
            cur_moves = moves.repeat(il, axis=0).cumsum(axis=0)
            one_out.extend((s_coord + cur_moves).tolist())
        one_out = np.array(one_out, dtype=int)
        return(one_out)
    
    def MakeHexMask(self, radius):
        f_ = np.zeros((radius*2*2+1, radius*2+1))
        indices = self.MakeHexMaskIndices(radius)
        for ind in indices:
            f_[tuple(ind)]=1.0
        return(f_)
    
    def MakeVariableIndices(self, radius, num_outputs, num_inputs):
        indices = self.MakeHexMaskIndices(radius)
        coords = -1*np.ones((indices.shape[0]*num_outputs*num_inputs,4),dtype=int) 
        coords[:,0:2]= np.tile(indices,(num_outputs*num_inputs,1))
        coords[:,2] = np.repeat(np.arange(num_inputs), indices.shape[0]*num_outputs)
        coords[:,3] = np.tile(np.repeat(np.arange(0,num_outputs),indices.shape[0]), num_inputs)
        return coords
    
    def MakeOutputMask(self, input_shape):
        radius= self.kernel_radius
        m_ = np.ones((input_shape[1] - 4*radius, input_shape[2] - 2*radius))
        for i in range(m_.shape[0]):
            for j in range(m_.shape[1]):
                m_[i,j] = ((i+j)%2)
        return(m_)    
    
    def build(self, input_shape):
        self.n_inputs = input_shape[-1]
        self.out_2D_shape = [input_shape[1] - 4*self.kernel_radius, input_shape[2]-2*self.kernel_radius]
        self.one_kernel_indices = self.MakeHexMaskIndices(self.kernel_radius)
        self.variable_indices = self.MakeVariableIndices(self.kernel_radius, self.num_outputs,self.n_inputs)
        #print(self.variable_indices.shape)
        #self.hex_mask = tf.constant( 
        #    self.MakeHexMask(self.kernel_radius).reshape([self.kernel_radius*4+1,self.kernel_radius*2+1,1,1]) ,
        #                            dtype=tf.float32 )
            
        mask_ = self.MakeHexMask(int((self.out_2D_shape[1]-1)/2))
        i_cut = int( 0.5*((mask_.shape[1]-1)/2*4 +1 -self.out_2D_shape[0] ))
        
        mask_ = mask_[i_cut:-i_cut,:]
        for coord in self.ignore_out:
            # somehow, giving tuple coordinates directly doesn't work when loading model
            mask_[tuple(coord)]= 0.0
        mask_ = tf.constant( mask_.reshape([self.out_2D_shape[0], 
                                            self.out_2D_shape[1], 
                                                        1]) , dtype=tf.float32)
        self.out_mask = mask_
        
        #self.ones_matrix = tf.constant(tf.ones([self.kernel_radius*4+1,
        #                                           self.kernel_radius*2+1,
        #                                           self.n_inputs,
        #                                           self.num_outputs]), dtype=tf.float32)
        
        self.sparse_weights = tf.Variable(self.winitializer(
                                                [self.variable_indices.shape[0]]
                                          ), 
                                          name="sparse_weights", dtype=tf.float32)
        self.offset = tf.Variable(tf.zeros([1,1,1,self.num_outputs]) , trainable=True, 
                                                    name="b_kernel", dtype=tf.float32)
        
        if type(self.activation)==str:
            self.activation = keras.activations.get(self.activation)
    def get_kernels(self):
        return tf.sparse.to_dense(tf.sparse.reorder(tf.SparseTensor(self.variable_indices,
                                     self.sparse_weights,
                                      dense_shape=[self.kernel_radius*4+1,
                                                   self.kernel_radius*2+1,
                                                   self.n_inputs,
                                                   self.num_outputs]
                                     )))
    def call(self, input):
        self.conv = tf.keras.backend.conv2d(input, 
                        tf.sparse.to_dense(
                            tf.sparse.reorder(
                               tf.SparseTensor(self.variable_indices,
                                               self.sparse_weights,
                                               dense_shape=[self.kernel_radius*4+1,
                                                           self.kernel_radius*2+1,
                                                           self.n_inputs,
                                                           self.num_outputs])
                            )
                        )
                    )
        return self.activation( self.conv + self.offset )*self.out_mask
