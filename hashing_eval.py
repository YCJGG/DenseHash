import numpy as np

def cat_ap_topK(cateTrainTest, HammingRank, M_set):
    precision = np.zeros((len(M_set), 1))
    recall = np.zeros((len(M_set), 1))

    numTest = cateTrainTest.shape[1]

    for i, K in enumerate(M_set):
        precisions = np.zeros((numTest, 1))
        recalls = np.zeros((numTest, 1))

        topk = HammingRank[:K, :]

        for qid in range(numTest):
            retrieved = topk[:, qid]
            rel = cateTrainTest[retrieved, qid]
            retrieved_relevant_num = np.sum(rel)
            real_relevant_num = np.sum(cateTrainTest[:, qid])

            precisions[qid] = retrieved_relevant_num/(K*1.0)
            recalls[qid] = retrieved_relevant_num/(real_relevant_num*1.0)

        precision[i] = np.mean(precisions)
        recall[i] = np.mean(recalls)

    return precision, recall

def cat_apcal(cateTrainTest, IX, num_return_NN=None):
    numTrain, numTest = IX.shape

    if num_return_NN: num_return_NN = numTrain

    apall = np.zeros((numTest, 1))

    for qid in range(numTest):
        query = IX[:, qid]
        x, p = 0, 0

        for rid in range(numTrain):
            if cateTrainTest[query[rid], qid]:
                x += 1.0
                p += x/(rid*1.0+1.0)

        if not p: apall[qid] = 0.0
        else: apall[qid] = p/(x*1.0)

    return np.mean(apall)

def evaluate_macro(Rel, Ret):
    '''
        evaluate macro_averaged performance
        Args:
            --Rel: relevant train documents for each test document
            --Ret: retrieved train documents for each test document
        Return:
            --p: macro-averaged precision
            --r: macro-averaged recall
    '''
    _, numTest = Rel.shape
    precisions = np.zeros(shape=[1, numTest], dtype=float)
    recalls = np.zeros(shape=[1, numTest], dtype=float)

    retrieved_relevant_pairs = (Rel & Ret)

    for j in range(0, numTest):
        retrieved_relevant_num = np.sum(retrieved_relevant_pairs[:,j])
        retrieved_num = np.sum(Ret[:,j])
        relevant_num = np.sum(Rel[:,j])
        if retrieved_num:
            precisions[0, j] = float(retrieved_relevant_num) / retrieved_num
        else:
            precisions[0, j] = 0

        if relevant_num:
            recalls[0, j] = float(retrieved_relevant_num) / relevant_num
        else:
            recalls[0, j] = 0

    p = np.mean(precisions)
    r = np.mean(recalls)
    return p, r
