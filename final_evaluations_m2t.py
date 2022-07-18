from datetime import datetime
from text_loaders.dataset_text_loader import get_dataset_text_loader
from text_loaders.model_text_loaders import get_text_loader
from utils.get_opt import get_opt
from utils.metrics import *
from networks.evaluator_wrapper import EvaluatorModelWrapper
from collections import OrderedDict
from utils.plot_script import *
from scripts.motion_process import *
from utils.utils import *
from bert_score import score
from nlgeval import NLGEval
from os.path import join as pjoin

import logging
import transformers
transformers.tokenization_utils.logger.setLevel(logging.ERROR)
transformers.configuration_utils.logger.setLevel(logging.ERROR)
transformers.modeling_utils.logger.setLevel(logging.ERROR)


def plot_t2m(data, save_dir, captions):
    data = gt_dataset.inv_transform(data)
    # print(ep_curves.shape)
    for i, (caption, joint_data) in enumerate(zip(captions, data)):
        joint = recover_from_ric(torch.from_numpy(joint_data).float(), wrapper_opt.joints_num).numpy()
        save_path = pjoin(save_dir, '%02d.mp4'%(i))
        plot_3d_motion(save_path, paramUtil.t2m_kinematic_chain, joint, title=caption, fps=20)
        # print(ep_curve.shape)

torch.multiprocessing.set_sharing_strategy('file_system')

def evaluate_matching_score(text_loaders, file):
    match_score_dict = OrderedDict({})
    R_precision_dict = OrderedDict({})
    # print(text_loaders.keys())
    print('========== Evaluating Matching Score ==========')
    for text_loader_name, text_loader in text_loaders.items():
        all_size = 0
        matching_score_sum = 0
        top_k_count = 0
        # print(text_loader_name)
        with torch.no_grad():
            for idx, batch in enumerate(text_loader):
                word_embeddings, pos_one_hots, _, sent_lens, motions, _, m_lens, _ = batch
                # word_embeddings, pos_one_hots, _, sent_lens, motions, m_lens, _ = batch
                text_embeddings, motion_embeddings = eval_wrapper.get_co_embeddings(
                    word_embs=word_embeddings,
                    pos_ohot=pos_one_hots,
                    cap_lens=sent_lens,
                    motions=motions,
                    m_lens=m_lens
                )
                dist_mat = euclidean_distance_matrix(motion_embeddings.cpu().numpy(),
                                                     text_embeddings.cpu().numpy())
                matching_score_sum += dist_mat.trace()

                argsmax = np.argsort(dist_mat, axis=1)
                top_k_mat = calculate_top_k(argsmax, top_k=3)
                top_k_count += top_k_mat.sum(axis=0)

                all_size += text_embeddings.shape[0]

            matching_score = matching_score_sum / all_size
            R_precision = top_k_count / all_size
            match_score_dict[text_loader_name] = matching_score
            R_precision_dict[text_loader_name] = R_precision

        print(f'---> [{text_loader_name}] Matching Score: {matching_score:.4f}')
        print(f'---> [{text_loader_name}] Matching Score: {matching_score:.4f}', file=file, flush=True)

        line = f'---> [{text_loader_name}] R_precision: '
        for i in range(len(R_precision)):
            line += '(top %d): %.4f ' % (i+1, R_precision[i])
        print(line)
        print(line, file=file, flush=True)

    return match_score_dict, R_precision_dict

def _strip(s):
    return s.strip()

def evaluate_bleu_rouge_cider(text_loaders, file):
    bleu_score_dict = OrderedDict({})
    rouge_score_dict = OrderedDict({})
    cider_score_dict = OrderedDict({})
    # print(text_loaders.keys())
    print('========== Evaluating NLG Score ==========')
    for text_loader_name, text_loader in text_loaders.items():

        ref_list = [list(refs) for refs in zip(*text_loader.dataset.all_caption_list)]
        cand_list = text_loader.dataset.generated_texts_list
        scores = nlg_eval.compute_metrics(ref_list, cand_list)
        bleu_score_dict[text_loader_name] = np.array([scores['Bleu_1'],scores['Bleu_2'],scores['Bleu_3'],scores['Bleu_4']])
        rouge_score_dict[text_loader_name] = scores['ROUGE_L']
        cider_score_dict[text_loader_name] = scores['CIDEr']

        line = f'---> [{text_loader_name}] BLEU: '
        for i in range(4):
            line += '(%d): %.4f ' % (i + 1, scores['Bleu_%d'%(i + 1)])
        print(line)
        print(line, file=file, flush=True)

        print(f'---> [{text_loader_name}] ROUGE_L: {scores["ROUGE_L"]:.4f}')
        print(f'---> [{text_loader_name}] ROUGE_L: {scores["ROUGE_L"]:.4f}', file=file, flush=True)
        print(f'---> [{text_loader_name}] CIDER: {scores["CIDEr"]:.4f}')
        print(f'---> [{text_loader_name}] CIDER: {scores["CIDEr"]:.4f}', file=file, flush=True)
    return bleu_score_dict, rouge_score_dict, cider_score_dict


def evaluate_bert_score(text_loaders, file):
    bert_score_dict = OrderedDict({})
    print('========== Evaluating BERT Score ==========')
    for text_loader_name, text_loader in text_loaders.items():
        P, R, F1 = score(text_loader.dataset.generated_texts_list,
                         text_loader.dataset.all_caption_list,
                         lang='en',
                         rescale_with_baseline=True,
                         idf=True,
                         device=device,
                         verbose=True)
        bert_score_dict[text_loader_name] = F1.mean().item()

        print(f'---> [{text_loader_name}] BERT SCORE: {F1.mean().item():.4f}')
        print(f'---> [{text_loader_name}] BERT SCORE: {F1.mean().item():.4f}', file=file, flush=True)
    return bert_score_dict


def get_metric_statistics(values):
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    conf_interval = 1.96 * std / np.sqrt(replication_times)
    return mean, conf_interval


def evaluation(log_file):
    all_metrics = OrderedDict({'Matching Score': OrderedDict({}),
                               'R_precision': OrderedDict({})
                               })

    with open(log_file, 'w') as f:
        text_loaders = {}

        for text_loader_name, text_loader_getter in eval_text_loaders.items():
            text_loader = text_loader_getter()
            text_loaders[text_loader_name] = text_loader

        print(f'Time: {datetime.now()}')
        print(f'Time: {datetime.now()}', file=f, flush=True)
        bleu_score_dict, rouge_score_dict, cider_score_dict = evaluate_bleu_rouge_cider(text_loaders, f)

        print(f'Time: {datetime.now()}')
        print(f'Time: {datetime.now()}', file=f, flush=True)
        bert_score_dict = evaluate_bert_score(text_loaders, f)

        # text_loaders = {}
        for replication in range(replication_times):
            # if replication == 0:
                # for text_loader_name, text_loader_getter in eval_text_loaders.items():
                #     text_loader = text_loader_getter()
                #     text_loaders[text_loader_name] = text_loader
            text_loaders['ground truth'] = gt_loader
            print(f'==================== Replication {replication} ====================')
            print(f'==================== Replication {replication} ====================', file=f, flush=True)
            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)

            mat_score_dict, R_precision_dict = evaluate_matching_score(text_loaders, f)
            print(f'!!! DONE !!!')
            print(f'!!! DONE !!!', file=f, flush=True)

            for key, item in mat_score_dict.items():
                if key not in all_metrics['Matching Score']:
                    all_metrics['Matching Score'][key] = [item]
                else:
                    all_metrics['Matching Score'][key] += [item]

            for key, item in R_precision_dict.items():
                if key not in all_metrics['R_precision']:
                    all_metrics['R_precision'][key] = [item]
                else:
                    all_metrics['R_precision'][key] += [item]

        for metric_name, metric_dict in all_metrics.items():
            print('========== %s Summary ==========' % metric_name)
            print('========== %s Summary ==========' % metric_name, file=f, flush=True)

            for model_name, values in metric_dict.items():
                # print(metric_name, model_name)
                mean, conf_interval = get_metric_statistics(np.array(values))
                # print(mean, mean.dtype)
                if isinstance(mean, np.float64) or isinstance(mean, np.float32):
                    print(f'---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}')
                    print(f'---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}', file=f, flush=True)
                elif isinstance(mean, np.ndarray):
                    line = f'---> [{model_name}]'
                    for i in range(len(mean)):
                        line += '(top %d) Mean: %.4f CInt: %.4f;' % (i+1, mean[i], conf_interval[i])
                    print(line)
                    print(line, file=f, flush=True)


def animation_4_user_study(save_dir):
    text_loaders = {}
    # mm_motion_loaders = {}
    for text_loader_name, text_loader_getter in eval_text_loaders.items():
        text_loader = text_loader_getter()
        text_loaders[text_loader_name] = text_loader

    text_loaders['ground_truth'] = gt_loader
    for text_loader_name, text_loader in text_loaders.items():
        for idx, batch in enumerate(text_loader):
            # if idx > 10:
            #     break
            word_embeddings, pos_one_hots, captions, sent_lens, motions, m_lens, tokens = batch
            motions = motions[:, :m_lens[0]]
            # plot_t2m(motions.cpu().numpy(), save_path, captions)
            print('-----%d-----'%idx)
            print(captions)
            print(tokens)
            print(sent_lens)
            print(m_lens)
            ani_save_path = pjoin(save_dir, 'animation', '%02d'%(idx))
            joint_save_path = pjoin(save_dir, 'keypoints', '%02d'%(idx))
            os.makedirs(ani_save_path, exist_ok=True)
            os.makedirs(joint_save_path, exist_ok=True)

            data = gt_dataset.inv_transform(motions[0])
            # print(ep_curves.shape)
            joint = recover_from_ric(data.float(), wrapper_opt.joints_num).cpu().numpy()
            joint = motion_temporal_filter(joint)
            np.save(pjoin(joint_save_path, text_loader_name+'.npy'), joint)
            # save_path = pjoin(save_dir, '%02d.mp4' % (idx))
            plot_3d_motion(pjoin(ani_save_path, '%s.mp4' % (text_loader_name)),
                           paramUtil.t2m_kinematic_chain, joint, title=captions[0], fps=20)


if __name__ == '__main__':
    dataset_opt_path = './checkpoints/t2m/Comp_v6_KLD005/opt.txt'
    eval_text_loaders = {
        # For HumanML3D dataset
        'M2T_EL4_DL4_NH8_PS': lambda: get_text_loader(
            './checkpoints/t2m/M2T_EL4_DL4_NH8_PS/opt.txt',
            batch_size, gt_dataset, device
        ),

        # For KIT-ML dataset

        # 'M2T_EL3_DL3_NH8_PS': lambda: get_text_loader(
        #     './checkpoints/kit/M2T_EL3_DL3_NH8_PS/opt.txt',
        #     batch_size, gt_dataset, device
        # ),
    }

    device_id = 0
    device = torch.device('cuda:%d'%device_id if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device_id)
    # device = "cpu"

    nlg_eval = NLGEval(
        metrics_to_omit=['METEOR',
                         'EmbeddingAverageCosineSimilarity' ,
                         'SkipThoughtCS',
                         'VectorExtremaCosineSimilarity',
                         'GreedyMatchingScore']
    )

    replication_times = 1
    batch_size = 32

    gt_loader, gt_dataset = get_dataset_text_loader(dataset_opt_path, batch_size, device)
    wrapper_opt = get_opt(dataset_opt_path, device)
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    log_file = './m2t_evaluation_t2m.log'
    evaluation(log_file)
    # animation_4_user_study('./user_study3/')