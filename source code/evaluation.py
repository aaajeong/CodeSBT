import nltk
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.meteor_score import meteor_score
nltk.download('wordnet')
nltk.download('omw-1.4')

def nltk_sentence_bleu(hypotheses, references, order=4):
    refs = []
    count = 0
    total_score = 0.0
    cc = SmoothingFunction()
    for hyp, ref in zip(hypotheses, references):
        hyp = hyp.split()
        ref = ref.split()
        refs.append([ref])
        if len(hyp) < order:
            continue
        else:
            score = nltk.translate.bleu([ref], hyp, smoothing_function=cc.method4)
            total_score += score
            count += 1
    avg_score = total_score / count
    print('avg_score: %.4f' % avg_score)
    return avg_score

# def nltk_sentence_bleu(hypotheses, references, order=4):
#     refs = []
#     count = 0
#     total_score = 0.0
#     # cc = SmoothingFunction()
#     for hyp, ref in zip(hypotheses, references):
#         hyp = hyp.split()
#         ref = ref.split()
#         refs.append([ref])
#         if len(hyp) < order:
#             continue
#         else:
#             # score = nltk.translate.bleu([ref], hyp, smoothing_function=cc.method4)
#             score = meteor_score([ref], hyp)
#             total_score += score
#             count += 1
#     avg_score = total_score / count
#     print('avg_score: %.4f' % avg_score)
#     return avg_score