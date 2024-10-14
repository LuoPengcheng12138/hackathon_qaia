"""online judger for QAIA-MLD problem"""
import pickle
from glob import glob
from typing import List, Tuple
import numpy as np
import os
import pdb
gpu_num = 0 # Use "" to use the CPU
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from sionna.mimo import lmmse_equalizer
from sionna.mapping import SymbolDemapper, Mapper, Demapper
# Import Sionna
try:
    import sionna as sn
except ImportError as e:
    # Install Sionna if package is not already installed
    import os
    os.system("pip install sionna")
    import sionna as sn
from sionna.channel import FlatFadingChannel, KroneckerModel
# Import TensorFlow and NumPy
import tensorflow as tf
# Avoid warnings from TensorFlow
tf.get_logger().setLevel('ERROR')
import numpy as np

# For plotting

# also try %matplotlib widget

import matplotlib.pyplot as plt

# for performance measurements 
import time

# For the implementation of the Keras models
from tensorflow.keras import Model

'''
本文件禁止改动!!!
'''

def compute_ber(solution, bits):
    '''
    Compute BER for the solution from QAIAs.

    Firstly, both the solution from QAIAs and generated bits should be transformed into gray-coded, 
    and then compute the ber.

    Reference
    ---------
    [1] Kim M, Venturelli D, Jamieson K. Leveraging quantum annealing for large MIMO processing in centralized radio access networks. 
        Proceedings of the ACM special interest group on data communication. 2019: 241-255.\
    
    Input
    -----
    solution: [2*Nt, ], np.int
        The binary array filled with ones and minus one.

    bits: [2*Nt, ], np.int
        The binary array filled with ones and minus one.
    Ouput
    -----
    ber: np.float
        A scalar, the BER.
    '''
    ## convert the bits from sionna style to constellation style
    #print("bits",bits[0:6,:])

    bits_constellation = 1 - np.concatenate([bits[..., 0::2], bits[..., 1::2]], axis=-1)# 奇数列与偶数列分别拼接 并取反 0<->1
    #print("bits_constellation",bits_constellation[0:6,:])

    num_bits_per_symbol = bits_constellation.shape[1]
    ## convert QuAMax transform to gray coded
    rb = num_bits_per_symbol//2
    bits_hat = solution.reshape(rb, 2, -1) #rb行 2列 每个格子64个元素
    bits_hat = np.concatenate([bits_hat[:, 0], bits_hat[:, 1]], 0) # 奇数列与偶数列分别拼接 rb*2行 64列
    bits_hat = bits_hat.T.copy()#  64行 rb*2（bits_per_symbol）列
    # convert Ising {-1, 1} to QUBO {0, 1}
    bits_hat[bits_hat == -1] = 0
    # Differential bit encoding
    index = np.nonzero(bits_hat[:, rb-1] == 1)[0] #rb-1列的非0元素的下标
    output_bit = bits_hat.copy()
    bits_hat[index, rb:] = 1 - bits_hat[index, rb:] #对rb到最后一列的（即一半列元素）0<->1
    #print("bits_hat",bits_hat[0:6,:])
    for i in range(num_bits_per_symbol - 1):
        output_bit[:, i + 1] = np.logical_xor(bits_hat[:, i], bits_hat[:, i + 1]).astype(np.float32)
    ber = np.mean(bits_constellation != output_bit)
    return ber


class Judger:
    """Judge contestant's algorithm with MLD test cases."""
    def __init__(self, test_cases):
        self.test_cases = test_cases

    @staticmethod
    def infer(ising_generator, qaia_mld_solver, H, y, num_bits_per_symbol, snr):
        J, h = ising_generator(H, y, num_bits_per_symbol, snr)
        bits = qaia_mld_solver(J, h)
        return bits
    
    def benchmark(self, ising_gen, qaia_mld_solver):
        avgber = 0
        for case in self.test_cases:
            H, y, bits_truth, num_bits_per_symbol, snr = case
            #print("bits_truth",bits_truth[0:15,:])
            bits_decode = self.infer(ising_gen, qaia_mld_solver, H, y, num_bits_per_symbol, snr)
            # print("bits_decode",bits_decode)
            ber = compute_ber(bits_decode, bits_truth)
            avgber += ber
        avgber /= len(self.test_cases)
        return avgber

    def bits_to_IQ(self):
        for case in self.test_cases:
            H, y, bits_truth, my_num_bits_per_symbol, snr = case
            #pdb.set_trace()
            constellation = sn.mapping.Constellation("qam", my_num_bits_per_symbol)
            constellation.show()
            mapper = sn.mapping.Mapper(constellation=constellation)

            no = sn.utils.ebnodb2no(ebno_db=snr,
                        num_bits_per_symbol=my_num_bits_per_symbol,
                        coderate=1.0) # Coderate num_bits_per_symbol to 1 as we do uncoded transmission here     
            # print("no: ", no)   
            # print("Shape of bits: ", bits_truth.shape)
            x = mapper(bits_truth)  #复数
            print("x:",x)
            # cal_y=H@x
            # print("cal_y:",cal_y[0:10])
            # print("H:",H)
            # channel = FlatFadingChannel(64, 64, add_awgn=True, return_channel=True)
            # yy,h = channel([x, no])
            # print("Shape of h: ", h.shape)
            # print("Shape of yy: ", yy.shape)


            channel = FlatFadingChannel(64, 64, add_awgn=True, return_channel=True)
            no = 0.01 # Noise variance of the channel

            x1 = tf.transpose(x)
            my_y, my_h = channel([x1, no])
            print("my_h",np.linalg.cond(my_h, p=None))
            print("H",np.linalg.cond(H, p=None))
            s = tf.cast(no*tf.eye(64, 64), y.dtype)
            x_hat, no_eff = lmmse_equalizer(my_y, my_h, s)
            cal_y=H@x

            plt.figure(figsize=(8,8))
            #plt.axes().set_aspect(1.0)
            plt.scatter(np.real(cal_y), np.imag(cal_y))
            plt.scatter(np.real(y), np.imag(y))
            plt.show()

            # s = tf.cast(no*tf.eye(64, 64), y.dtype)
            y= tf.convert_to_tensor(y)
            y = tf.transpose(y)
            print("y: ", y.shape)
            H= tf.convert_to_tensor(H)
            H = tf.expand_dims(H, 0)
            print("H: ", H)
            x_hat, no_eff = lmmse_equalizer(y, H,s)

            plt.figure(figsize=(8,8))
            #plt.axes().set_aspect(1.0)
            plt.scatter(np.real(x_hat), np.imag(x_hat))
            plt.scatter(np.real(x), np.imag(x))
            plt.show()

            # symbol_demapper = SymbolDemapper("qam", my_num_bits_per_symbol, hard_out=True)
            # x_ind = symbol_demapper([x, no])
            # print("x_ind=\n",x_ind)
            # x_ind_hat = symbol_demapper([x_hat, no])
            # print("x_ind_hat=\n",x_ind_hat)


            # I= 2*(bits_truth[:,0]*2+bits_truth[:,1])-3
            # Q= 2*(bits_truth[:,2]*2+bits_truth[:,3])-3
            # IQ = (I+1j*Q).reshape(64,1) # 奇数列与偶数列分别拼接 rb*2行 64列
            # print("IQ=\n",IQ[0:15])

            # print("res=\n",(H@x)[0:15])
            # print("y=\n",y[0:15])

            # plt.figure(figsize=(8,8))
            # plt.axes().set_aspect(1)
            # plt.grid(True)
            # plt.title('Channel output')
            # plt.xlabel('Real Part')
            # plt.ylabel('Imaginary Part')
            # plt.scatter(tf.math.real(y), tf.math.imag(y))
            # plt.tight_layout()
            # plt.show()



if __name__ == "__main__":
    from main import ising_generator, qaia_mld_solver
    file_path = 'MLD_data/'
    filelist = glob(f'{file_path}0.pickle')
    dataset = []
    for filename in filelist:
        # 读取数据
        data = pickle.load(open(filename, 'rb'))
        dataset.append([data['H'], data['y'], data['bits'], data['num_bits_per_symbol'], data['SNR']])

    judger = Judger(dataset)
    judger.bits_to_IQ()
    avgber = judger.benchmark(ising_generator, qaia_mld_solver)
    # 测试选手的平均ber，越低越好
    print(f"score: {avgber}")