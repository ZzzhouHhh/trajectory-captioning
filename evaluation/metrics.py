"""
轨迹描述评测指标计算。

评测指标:
- BLEU: n-gram precision (BLEU-1, BLEU-2)
- ROUGE: n-gram recall (ROUGE-1, ROUGE-2, ROUGE-L)
- BERTScore: semantic similarity via microsoft/deberta-xlarge-mnli (F1)

用法:
    from evaluation import evaluate_all
    results = evaluate_all(generated_captions, reference_captions)
"""

import numpy as np
from collections import Counter
import math


# ========== BLEU ==========

def _ngrams(tokens, n):
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def compute_bleu(references, candidates, max_n=2):
    """
    计算 BLEU-1 和 BLEU-2 分数。

    BLEU = BP · exp(Σ w_n · log p_n)
    BP = min(1, exp(1 - r/c))   (短句惩罚)
    w_n = 1/N (均匀权重)

    返回: {'BLEU-1': float, 'BLEU-2': float}
    """
    bleu_scores = {}

    for n in range(1, max_n + 1):
        precisions = []
        bp_sum = 0.0
        count = 0

        for refs, cand in zip(references, candidates):
            if isinstance(refs, str):
                refs = [refs]

            cand_tokens = cand.lower().split()
            c = len(cand_tokens)
            if c == 0:
                precisions.append(0.0)
                continue

            # 找最接近长度的参考句
            best_ref = min(refs, key=lambda r: abs(len(r.split()) - c))
            ref_tokens = best_ref.lower().split()
            r = len(ref_tokens)

            cand_ngrams = Counter(_ngrams(cand_tokens, n))
            ref_ngrams = Counter(_ngrams(ref_tokens, n))

            match_count = sum((cand_ngrams & ref_ngrams).values())
            total_count = max(len(cand_ngrams), 1)

            p_n = match_count / total_count
            precisions.append(p_n)

            # BP: brevity penalty
            if c >= r:
                bp = 1.0
            else:
                bp = math.exp(1 - r / c)
            bp_sum += bp
            count += 1

        avg_precision = np.mean(precisions) if precisions else 0.0
        avg_bp = bp_sum / count if count > 0 else 1.0
        bleu = avg_bp * avg_precision
        bleu_scores[f'BLEU-{n}'] = round(bleu, 4)

    return bleu_scores


# ========== ROUGE ==========

def _lcs(a, b):
    """Longest Common Subsequence length"""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def compute_rouge(references, candidates):
    """
    计算 ROUGE-1, ROUGE-2, ROUGE-L 分数。


    返回: {'ROUGE-1': float, 'ROUGE-2': float, 'ROUGE-L': float}
    """
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []

    for refs, cand in zip(references, candidates):
        if isinstance(refs, str):
            refs = [refs]

        cand_tokens = cand.lower().split()
        if len(cand_tokens) == 0:
            rouge1_scores.append(0.0)
            rouge2_scores.append(0.0)
            rougeL_scores.append(0.0)
            continue

        best_r1, best_r2, best_rL = 0.0, 0.0, 0.0

        for ref in refs:
            ref_tokens = ref.lower().split()

            # ROUGE-1
            ref_1grams = Counter(_ngrams(ref_tokens, 1))
            cand_1grams = Counter(_ngrams(cand_tokens, 1))
            overlap_1 = sum((ref_1grams & cand_1grams).values())
            r1 = overlap_1 / max(len(ref_1grams), 1)

            # ROUGE-2
            ref_2grams = Counter(_ngrams(ref_tokens, 2))
            cand_2grams = Counter(_ngrams(cand_tokens, 2))
            overlap_2 = sum((ref_2grams & cand_2grams).values())
            r2 = overlap_2 / max(len(ref_2grams), 1)

            # ROUGE-L
            lcs_len = _lcs(ref_tokens, cand_tokens)
            p_lcs = lcs_len / max(len(cand_tokens), 1)
            r_lcs = lcs_len / max(len(ref_tokens), 1)
            if p_lcs + r_lcs > 0:
                rL = 2 * p_lcs * r_lcs / (p_lcs + r_lcs)
            else:
                rL = 0.0

            best_r1 = max(best_r1, r1)
            best_r2 = max(best_r2, r2)
            best_rL = max(best_rL, rL)

        rouge1_scores.append(best_r1)
        rouge2_scores.append(best_r2)
        rougeL_scores.append(best_rL)

    return {
        'ROUGE-1': round(np.mean(rouge1_scores), 4) if rouge1_scores else 0.0,
        'ROUGE-2': round(np.mean(rouge2_scores), 4) if rouge2_scores else 0.0,
        'ROUGE-L': round(np.mean(rougeL_scores), 4) if rougeL_scores else 0.0,
    }


# ========== BERTScore ==========

def compute_bertscore(references, candidates, model_name='microsoft/deberta-xlarge-mnli'):
    """
    计算 BERTScore (Precision, Recall, F1)。


    如果 transformers 不可用，返回占位值。
    """
    try:
        from transformers import AutoTokenizer, AutoModel
        import torch

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        model.eval()

        precisions, recalls, f1s = [], [], []

        for refs, cand in zip(references, candidates):
            if isinstance(refs, str):
                refs = [refs]

            best_f1 = 0.0
            best_p = 0.0
            best_r = 0.0

            for ref in refs:
                with torch.no_grad():
                    cand_inputs = tokenizer(cand, return_tensors='pt',
                                            truncation=True, max_length=512).to(device)
                    ref_inputs = tokenizer(ref, return_tensors='pt',
                                           truncation=True, max_length=512).to(device)

                    cand_emb = model(**cand_inputs).last_hidden_state.mean(dim=1)
                    ref_emb = model(**ref_inputs).last_hidden_state.mean(dim=1)

                    cos_sim = torch.cosine_similarity(cand_emb, ref_emb).item()
                    # 简化版: 使用均值池化的余弦相似度
                    p = (cos_sim + 1) / 2  # 缩放到[0,1]
                    r = p
                    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

                if f1 > best_f1:
                    best_f1 = f1
                    best_p = p
                    best_r = r

            precisions.append(best_p)
            recalls.append(best_r)
            f1s.append(best_f1)

        return {
            'BERTScore-P': round(np.mean(precisions), 4),
            'BERTScore-R': round(np.mean(recalls), 4),
            'BERTScore-F1': round(np.mean(f1s), 4),
        }

    except ImportError:
        print("Warning: transformers not available, using placeholder BERTScore.")
        return {'BERTScore-P': 0.0, 'BERTScore-R': 0.0, 'BERTScore-F1': 0.0}


# ========== 综合评测 ==========

def evaluate_generated_captions(generated, references):
    """
    对单组生成结果和参考文本进行评测。

    参数:
        generated: list of str (生成文本列表)
        references: list of str or list of list of str (参考文本列表)

    返回:
        包含所有指标的字典
    """
    return {
        **compute_bleu(references, generated),
        **compute_rouge(references, generated),
        **compute_bertscore(references, generated),
    }


def evaluate_all(generated, references):
    """
    完整评测，打印结果表格。

    用法:
        results = evaluate_all(generated_captions, reference_captions)
    """
    results = evaluate_generated_captions(generated, references)

    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    for metric, value in results.items():
        print(f"  {metric:15s}: {value:.4f}")
    print("=" * 50)

    return results


# ========== GPT-4 评测 ==========
# 使用GPT-4作为自动评测指标之一（类似G-Eval），
# 该部分代码依赖于OpenAI API，在此作为可选功能。
#
# def compute_gpt4_score(generated, references, api_key=None):
#     """使用GPT-4对生成文本进行1-5评分"""
#     # 需要 openai package
#     pass
