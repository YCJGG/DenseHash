import DenseHash_RF as dh
import pickle
from datetime import datetime

def DenseHash_RF_demo():
    lamda = 50
    param = {}
    param['lambda'] = lamda

    gpu_ind = 0
    bits = [12]
    # bits = [12, 24, 32, 48]
    for bit in bits:
        filename = 'snapshot/denseHash_RF_nop4p5Stride_111_' + str(bit) + 'bits_CIFAR_10' + '_' + datetime.now().strftime("%y%m%d_%H%M") + '.pkl'
        param['filename'] = filename
        print('---------------------------------------')
        print('[#bit: %3d]' % (bit))
        result = dh.DenseHash_RF_algo(bit, param, gpu_ind)
        print('[MAP: %3.5f]' % (result['map']))
        print('---------------------------------------')
        #fp = open(result['filename'], 'wb')
        #pickle.dump(result, fp)
        #fp.close()

if __name__=="__main__":
    DenseHash_RF_demo()
