import os
import sys
import re
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm




def get_results_only(logfile, outfile):
    with open(logfile, 'r') as logfile:
        with open(outfile, 'a') as out:
            log = logfile.readlines()
            for l in log:
                if l.find("step") > 0 and l.find("epoch") > 0:
                    out.write(l + '\n')
                elif l.find("test eval") > 0:
                    out.write(l + '\n')
                elif l.find("test avg") > 0:
                    out.write(l + '\n')
                else:
                    continue

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
            if l.find("step") > 0 and l.find("epoch") > 0:
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
def get_graph_one(dataset, model):
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
    plt.xlim([0, 50])      # X축의 범위: [xmin, xmax]
    # plt.ylim([0, 1])     # Y축의 범위: [ymin, ymax]
    plt.title(title)

    plt.show()
    plt.savefig('./img/'+title+'.png')

def get_graph_all():
    route = os.path.abspath(os.path.join(os.curdir, os.pardir))
    deepcom = route+'/emse-data(ast_only)/model/default/log.txt'
    deepcom_p = route+'/emse-data(sbt_code)/model/default/log.txt'
    h_deepcom = route+'/emse-data(original)/model/hybrid/log.txt'
    h_deepcom_p= route+'/emse-data(sbt_code_hb)/model/hybrid/log.txt'
    
    dict_dp = get_training_results(deepcom)
    dict_dp2 = get_training_results(deepcom_p)
    dict_hdp = get_training_results(h_deepcom)
    dict_hdp2 = get_training_results(h_deepcom_p)


    # plt.plot(dict_dp['epoch'], dict_dp['test_avg_score'], label='deepcom')
    # plt.plot(dict_dp2['epoch'], dict_dp2['test_avg_score'], label='deepcom(our_data)')
    # plt.xlabel('Epoch')
    # plt.ylabel('Test Avg Score(BLEU)')
    # plt.title("Deepcom VS Deepcom(our model)")
    # plt.legend()

    # plt.show()
    # plt.savefig('./img/'+'Deepcom compare.png')

    plt.plot(dict_hdp['epoch'], dict_hdp['test_avg_score'], label='h-deepcom')
    plt.plot(dict_hdp2['epoch'], dict_hdp2['test_avg_score'], label='h-deepcom(our_data)')
    plt.xlabel('Epoch')
    plt.ylabel('Test Avg Score(BLEU)')
    plt.title("Hybrid-Deepcom VS Hybrid-Deepcom(our model)")
    plt.legend()

    plt.show()
    plt.savefig('./img/'+'Hybrid-Deepcom compare.png')

# get_graph_one('emse-data(sbt_code_hb)', 'hybrid')
# get_graph_all()
get_results_only('../emse-data(simsbt_code_hb)/model/hybrid/log.txt', 'simsbt_code_hb')