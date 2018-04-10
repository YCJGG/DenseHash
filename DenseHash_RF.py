from torchvision import transforms
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import models
import os
import numpy as np
import pickle
from datetime import datetime
import torch.nn.parallel

import utils.DataProcessing as DP
import utils.CalcHammingRanking as CalcHR

import CNN_model
import time 


def LoadLabel(filename, DATA_DIR):
    path = os.path.join(DATA_DIR, filename)
    fp = open(path, 'r')
    labels = [x.strip() for x in fp]
    fp.close()
    return torch.LongTensor(list(map(int, labels)))

def EncodingOnehot(target, nclasses):
    
    #print(target.shape)
    target_onehot = torch.FloatTensor(target.size(0), nclasses)

    target_onehot.zero_()
    target_onehot.scatter_(1, target.view(-1, 1), 1)
    return target_onehot

def CalcSim(batch_label, train_label):
    S = (batch_label.mm(train_label.t()) > 0).type(torch.FloatTensor)
    return S

def CreateModel(model_name, bit, use_gpu):
    if model_name == 'vgg16':
        vgg16 = models.vgg16(pretrained = True)
        #vgg16 = torch.load('/home/zhangjingyi/Rescode/vgg16_caffe2pytorch/vgg16_20M.pkl')
        cnn_model = CNN_model.cnn_model(vgg16, model_name, bit)
    print('**********************************')
    print('Def of poo4~poo5:')
    for i in [23,24,26,28,30]:
	print(cnn_model.features[i])
    print('**********************************')
    if use_gpu:
        cnn_model = torch.nn.DataParallel(cnn_model).cuda()
        #cnn_model = cnn_model.cuda()
    return cnn_model

def AdjustLearningRate(optimizer, epoch, learning_rate):
    lr = learning_rate * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def GenerateCode(model, data_loader, num_data, bit, use_gpu):
    B = np.zeros([num_data, bit], dtype=np.float32)
    for iter, data in enumerate(data_loader, 0):
        data_input, _, data_ind = data
        if use_gpu:
            data_input = Variable(data_input.cuda(), volatile=True)
        else: data_input = Variable(data_input, volatile=True)
        output,f = model(data_input)
        if use_gpu:
            B[data_ind.numpy(), :] = torch.sign(output.cpu().data).numpy()
        else:
            B[data_ind.numpy(), :] = torch.sign(output.data).numpy()
    return B,f

def DenseHash_RF_algo(bit, param, gpu_ind=0):
    # parameters setting
    #os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_indi)

    #os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

    #DATA_DIR = 'data/CIFAR-10'
    DATA_DIR = '/home/zhangjingyi/res/DPSH-pytorch/data/CIFAR-10'
    DATABASE_FILE = 'database_img.txt'
    TRAIN_FILE = 'train_img.txt'
    TEST_FILE = 'test_img.txt'

    DATABASE_LABEL = 'database_label.txt'
    TRAIN_LABEL = 'train_label.txt'
    TEST_LABEL = 'test_label.txt'

    batch_size = 64
    epochs = 30
    learning_rate = 0.003
    weight_decay = 10 ** -5
    model_name = 'vgg16'
    #model_name = 'alexnet'
    nclasses = 10
    use_gpu = torch.cuda.is_available()

    filename = param['filename']
    print('pkl file name %s'%filename)

    lamda = param['lambda']
    param['bit'] = bit
    param['epochs'] = epochs
    param['learning rate'] = learning_rate
    param['model'] = model_name
    print('**********************************')
    print('Solver Settings:')
    print(param)

    ### data processing
    transformations = transforms.Compose([
        transforms.Scale(256),
        #transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dset_database = DP.DatasetProcessingCIFAR_10(
        DATA_DIR, DATABASE_FILE, DATABASE_LABEL, transformations)

    dset_train = DP.DatasetProcessingCIFAR_10(
        DATA_DIR, TRAIN_FILE, TRAIN_LABEL, transformations)

    dset_test = DP.DatasetProcessingCIFAR_10(
        DATA_DIR, TEST_FILE, TEST_LABEL, transformations)

    num_database, num_train, num_test = len(dset_database), len(dset_train), len(dset_test)

    database_loader = DataLoader(dset_database,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=4
                             )

    train_loader = DataLoader(dset_train,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4
                             )

    test_loader = DataLoader(dset_test,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=4
                             )

    ### create model
    model = CreateModel(model_name, bit, use_gpu)

    optimizer = optim.SGD(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay)

    ### training phase
    # parameters setting
    B = torch.sign(torch.randn(num_train, bit))
    U = torch.sign(torch.randn(num_train, bit))
    train_labels = LoadLabel(TRAIN_LABEL, DATA_DIR)
    #print(train_labels)    

    train_labels_onehot = EncodingOnehot(train_labels, nclasses)
    test_labels = LoadLabel(TEST_LABEL, DATA_DIR)
    test_labels_onehot = EncodingOnehot(test_labels, nclasses)
    
    Y = train_labels_onehot
    #Sim = CalcSim(train_labels_onehot, train_labels_onehot)
    #file = open(filename.replace('snapshot/','log/').replace('.pkl','.log'),'a')
    for epoch in range(epochs):
	model.train()
        start_time = time.time()        
        epoch_loss = 0.0
        # D  step
        temp1 = Y.t().mm(Y)+ torch.eye(nclasses)
        temp1 = temp1.inverse()
        temp1 = temp1.mm(Y.t())
        D = temp1.mm(B)

        # B step
  
        B = torch.sign(Y.mm(D) + 1e-5 * U)
        print('[Epoch %3d B step time cost: %3.5fs]'%(epoch+1, time.time() - start_time))

        # F step
        ## training epoch
	ave_iter_loss = 0.0
        for iter, traindata in enumerate(train_loader, 0):
	    iter_timer = time.time()
            train_input, train_label, batch_ind = traindata
            train_input = Variable(train_input.cuda()) 
            model.zero_grad()
            train_outputs,f = model(train_input)
            #print(train_outputs.size())

            temp = torch.zeros(train_outputs.data.size())
            for i, ind in enumerate(batch_ind):
                temp[i,:] = B[ind, :]
                U[ind, :] = train_outputs.data[i]
            
            temp = Variable(temp.cuda())
            
            loss = (temp - train_outputs).pow(2).sum()/(batch_size)

           
            loss.backward()
            optimizer.step()
            epoch_loss += loss.data[0]
	    ave_iter_loss += loss.data[0]

	    if iter%20 == 0:
		print('[Iteration %d][%3.2fs/iter][Iter Loss: %3.5f]' % (iter+epoch*len(train_loader), time.time()-iter_timer, ave_iter_loss/20))
		ave_iter_loss = 0

        print('[Train Phase][Epoch: %3d/%3d][Loss: %3.5f]' % (epoch+1, epochs, epoch_loss / len(train_loader)))
        optimizer = AdjustLearningRate(optimizer, epoch, learning_rate)
       
        ### testing during epoch
	test_timer = time.time()
	model.eval()
        qB,f = GenerateCode(model, test_loader, num_test, bit, use_gpu)
        tB = torch.sign(U).numpy()
        f = f.cpu().data.numpy()
        np.save('f.npy',f)
        map_ = CalcHR.CalcMap(qB, tB, test_labels_onehot.numpy(), train_labels_onehot.numpy())
        print('[Test Phase ][Epoch: %3d/%3d] MAP(retrieval train): %3.5f' % (epoch+1, epochs, map_))
        map_topk = CalcHR.CalcTopMap(qB,tB,test_labels_onehot.numpy(), train_labels_onehot.numpy(),50)
        print('[Test Phase ][Epoch: %3d/%3d] MAP@top50(retrieval train): %3.5f' % (epoch+1, epochs, map_topk))
        map_topk = CalcHR.CalcTopMap(qB,tB,test_labels_onehot.numpy(), train_labels_onehot.numpy(),500)
        print('[Test Phase ][Epoch: %3d/%3d] MAP@top500(retrieval train): %3.5f' % (epoch+1, epochs, map_topk))
	acc_topk = CalcHR.CalcTopAcc(qB, tB, test_labels_onehot.numpy(), train_labels_onehot.numpy(), 50)
        print('[Test Phase ][Epoch: %3d/%3d] Precision@top50(retrieval train): %3.5f' % (epoch+1, epochs, acc_topk))
	acc_topk = CalcHR.CalcTopAcc(qB, tB, test_labels_onehot.numpy(), train_labels_onehot.numpy(), 500)
        print('[Test Phase ][Epoch: %3d/%3d] Precision@top500(retrieval train): %3.5f' % (epoch+1, epochs, acc_topk))
	print('[Test time cost: %d]'%(time.time() - test_timer))
        
        print('[Epoch %3d time cost: %ds]'%(epoch+1, time.time() - start_time))

        ### evaluation phase
        ## create binary code
        if (epoch + 1)%epochs == 0 : 
            eval_timer = time.time()   
            model.eval()
            database_labels = LoadLabel(DATABASE_LABEL, DATA_DIR)
            database_labels_onehot = EncodingOnehot(database_labels, nclasses)
            qB,_ = GenerateCode(model, test_loader, num_test, bit, use_gpu)
            dB,_ = GenerateCode(model, database_loader, num_database, bit, use_gpu)

            map = CalcHR.CalcMap(qB, dB, test_labels_onehot.numpy(), database_labels_onehot.numpy())
            print('[Retrieval Phase] MAP(retrieval database): %3.5f' % map)
            map_topk = CalcHR.CalcTopMap(qB, dB, test_labels_onehot.numpy(), database_labels_onehot.numpy(), 500)
            print('[Retrieval Phase] MAP@500(retrieval database): %3.5f' % map_topk)
            map_topk = CalcHR.CalcTopMap(qB, dB, test_labels_onehot.numpy(), database_labels_onehot.numpy(), 5000)
            print('[Retrieval Phase] MAP@5000(retrieval database): %3.5f' % map_topk)
	    acc_topk = CalcHR.CalcTopAcc(qB, dB, test_labels_onehot.numpy(), database_labels_onehot.numpy(), 500)
            print('[Retrieval Phase] Precision@500(retrieval database): %3.5f' % acc_topk)
	    acc_topk = CalcHR.CalcTopAcc(qB, dB, test_labels_onehot.numpy(), database_labels_onehot.numpy(), 5000)
            print('[Retrieval Phase] Precision@5000(retrieval database): %3.5f' % acc_topk)
            print('[Eval time: %ds]'%(time.time() - eval_timer))

    ## save trained model
    torch.save(model.state_dict(), filename)

    result = {}
    result['qB'] = qB
    result['dB'] = dB
    result['map'] = map

    return result

"""
if __name__=='__main__':
    bit = 12
    lamda = 50
    gpu_ind = 0
    filename = 'snapshot/denseHash_RF_nop4p5Stride_111_' + str(bit) + 'bits_CIFAR_10_' + datetime.now().strftime("%y%m%d-%H%M") + '.pkl'
    param = {}
    param['lambda'] = lamda
    param['filename'] = filename
    result = DPSH_algo(bit, param, gpu_ind)
    fp = open(result['filename'], 'wb')
    pickle.dump(result, fp)
    fp.close()
"""
