import os
gpu_num = 0 # Use "" to use the CPU
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import Sionna
try:
    import sionna as sn
except ImportError as e:
    # Install Sionna if package is not already installed
    import os
    os.system("pip install sionna")
    import sionna as sn

# Import TensorFlow and NumPy
import tensorflow as tf
# Avoid warnings from TensorFlow
tf.get_logger().setLevel('ERROR')
import numpy as np

# For plotting
# also try %matplotlib widget
import matplotlib
import matplotlib.pyplot as plt

# for performance measurements 
import time

# For the implementation of the Keras models
from tensorflow.keras import Model



# NUM_BITS_PER_SYMBOL = 4 # QPSK
# constellation = sn.mapping.Constellation("qam", NUM_BITS_PER_SYMBOL)

# constellation.show()

from sionna.mapping import MapperDemapper
from sionna.utils import BinarySource
# 定义64-QAM调制器
mapper = Mapper("qam",num_bits_per_symbol=6,modulation_order=64)
#生成比特流bit source = BinarySource()bits = bit source([100])#生成108个比特
# 调制
symbols = mapper(bits)
# 打印星座点
print(symbols)
# 生成星座图
plt.figure(figsize=(6，6))
plt.scatter(symbols.numpy().realsymbols.numpy().imag，color='blue')plt.title('64-QAM ConstellationDiagram')
plt.xlabel('In-phase')
plt.ylabel('Quadrature')plt.grid(True)plt.axhline(0，color='black'linewidth=0.5)color='black'plt.axvline(0,linewidth=0.5)plt.show()




# SNR=20
# Pnoise = 1/10**(SNR/10)
# Nt=64
# Nr=64
# #误码比特数
# X=np.random.choice([-3,-1, 1,3], size=(Nt, 1))+ 1j *np.random.choice([-3,-1, 1,3], size=(Nt, 1))
# # 定义信道矩阵的大小
# # 生成Rayleigh衰落信道矩阵
# matrix_size = (Nr,Nt)
# H = 1/np.sqrt(Nt) * (np.random.randn(*matrix_size) + 1j * np.random.randn(*matrix_size))/np.sqrt(2)
# n = np.sqrt(Pnoise/2) * (np.random.randn(Nr,1) + 1j * np.random.randn(Nr,1))

# y = np.dot(H,X) + n

# #共轭转置矩阵
# conjugate_transpose = np.conjugate(H.T)

# #zero forcing
# #inverse_matrix = np.linalg.inv(H)
# x_hat=np.dot(np.dot(np.linalg.inv(np.dot(conjugate_transpose,H)),conjugate_transpose),y)
# print("X=\n",X[0:4])
# print("y=\n",y[0:4])
# print("x_hat=\n",x_hat[0:4])
