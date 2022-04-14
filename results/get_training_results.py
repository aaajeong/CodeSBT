import os
import sys
import re
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm




# Get training result detail from log file
# step, epoch, learning rate, step-time, loss 7.814, test eval: loss 34.15
# test avt_score
def get_training_results(logfile):
    result_dict = {}
    step =[]
    epoch = []
    lr = []
    st = []
    loss = []
    test_eval_loss = []
    test_avg_score = []

    with open(logfile, 'r') as logfile:
        log = logfile.readlines()
        for l in tqdm(log):
            if l.find("step") > 0:
                step.append(re.search(r"(?<=step ).*?(?= epoch)", l).group(0))
                epoch.append(re.search(r"(?<=epoch ).*?(?= learning rate)", l).group(0))
                lr.append(re.search(r"(?<=learning rate ).*?(?= step-time)", l).group(0))
                st.append(re.search(r"(?<=step-time ).*?(?= loss)", l).group(0))
                loss.append(re.search(r"(?<=loss ).*", l).group(0))
            elif l.find("test eval") > 0:
                test_eval_loss.append(re.search(r"(?<=loss ).*", l).group(0))
            elif l.find("test avg_score") > 0:
                test_avg_score.append(re.search(r"(?<=score=).*", l).group(0))
            else:
                continue
    
    result_dict['step']=step
    result_dict['epoch']=epoch
    result_dict['lr']=lr
    result_dict['st']=st
    result_dict['loss']=loss
    result_dict['test_eval_loss']=test_eval_loss
    result_dict['test_avg_score']=test_avg_score

    return result_dict


# get graph for model 
# test evaluation loss, test avgerage score(blue) per step & epoch
# Input: dataset foler name, model
def get_graph(dataset, model):
    title = dataset
    log_file = os.path.abspath(os.path.join(os.curdir, os.pardir))+'/'+dataset+'/model/'+model+'/log.txt'
    result_dict = get_training_results(log_file)

    step = result_dict['step']
    epoch = result_dict['epoch']
    # eval_loss = result_dict['test_eval_loss'] 
    avg_blue = result_dict['test_avg_score']

    plt.plot(epoch, avg_blue)
    plt.xlabel('Epoch')
    plt.ylabel('Test Avg Score(BLEU)')
    plt.title(title)

    plt.show()
    plt.savefig('./img/'+title+'.png')



get_graph('graph_test', 'default')