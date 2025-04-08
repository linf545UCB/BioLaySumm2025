import os
import json
import argparse
import random
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
from scipy.spatial.distance import cosine
from gritlm import GritLM
import textstat


def gritlm_instruction(instruction):
    return "<|user|>\n" + instruction + "\n<|embed|>\n" if instruction else "<|embed|>\n"

def encode_sentences(model, sentences, instruction):
    return model.encode(sentences, instruction=gritlm_instruction(instruction))

def compute_cosine_similarity(model, references, candidates, instruction):
    ref_sents = sent_tokenize(references)
    can_sents = sent_tokenize(candidates)


    assert len(ref_sents) == len(can_sents), "Mismatch in number of sentences."

    ref_reps = encode_sentences(model, ref_sents, instruction)
    can_reps = encode_sentences(model, can_sents, instruction)

    
    scores = []
    for i in range(len(ref_sents)):
        score = 1 - cosine(ref_reps[i], can_reps[i])
        scores.append(score)
    return scores

def evaluate(prediction_file, groundtruth_file, task_name):
    # Load predictions and references
    with open(prediction_file, 'r', encoding='utf-8') as pred_f:
        predictions = json.load(pred_f)
    with open(groundtruth_file, 'r', encoding='utf-8') as gt_f:
        groundtruths = json.load(gt_f)

    model = GritLM("GritLM/GritLM-7B", torch_dtype="auto")
    instruction = "encode the medical text:"

    rouge = Rouge()
    chencherry = SmoothingFunction()

    samples = []
    for i, (pred, gt) in enumerate(zip(predictions, groundtruths)):
        if i % 100 == 0:
            print(f"Processing {i}")

        reference_text = gt["reference"]
        document_text = gt["document"]
        candidate_text = pred["generated_caption"]

        # Cosine similarity per sentence
        try:
            cosine_scores = compute_cosine_similarity(model, reference_text, candidate_text, instruction)
        except Exception as e:
            print(f"Skipping cosine for index {i} due to error: {e}")
            cosine_scores = []

        # BLEU
        references = [word_tokenize(reference_text)]
        candidates = word_tokenize(candidate_text)

        bleu_scores = [
            sentence_bleu(references, candidates, weights=(1, 0, 0, 0), smoothing_function=chencherry.method1),
            sentence_bleu(references, candidates, weights=(0.5, 0.5, 0, 0), smoothing_function=chencherry.method1),
            sentence_bleu(references, candidates, weights=(0.33, 0.33, 0.33, 0), smoothing_function=chencherry.method1),
            sentence_bleu(references, candidates, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=chencherry.method1)
        ]

        # METEOR
        meteor = meteor_score([references[0]], candidates)

        # ROUGE
        try:
            rouge_scores = rouge.get_scores(candidate_text, reference_text, avg=True)
        except:
            rouge_scores = {}
        
        if task_name == 'Lay_Summarisation':
            fkgl = textstat.flesch_kincaid_grade(candidate_text) #Flesch-Kincaid Grade Level (FKGL)
            dcrs = textstat.dale_chall_readability_score(candidate_text) #Dale-Chall Readability Score (DCRS)
            cli = textstat.coleman_liau_index(candidate_text) #Coleman-Liau Index (CLI)
            # r = Readability(candidate_text) #LENS 这个指标要确认一下
            # lens = r.linsear_write()
        #     alignScoreCS.score(context=reference_text, claim=candidate_text)# AlignScore
        #     SummaC = SummaCZS_model.score([document_text], [candidate_text])["scores"][0] #SummaC
            
        # elif task_name == 'Radiology_Report_Generation':
        #     accuracy, accuracy_not_averaged, class_report, class_report_5 = f1chexbert(hyps=[candidate_text], refs=[reference_text])
        #     mean_reward, reward_list, hypothesis_annotation_lists, reference_annotation_lists = f1radgraph(hyps=[candidate_text], refs=[reference_text]) #f1radgraph
        #     #RadCliQ

        if task_name == 'Lay_Summarisation':
            samples.append({
                "reference": reference_text,
                "generated_caption": candidate_text,
                "BLEU": bleu_scores,
                "meteor": meteor,
                "rouge": rouge_scores,
                "cosine_similarity": cosine_scores,
                "fkgl": fkgl,
                "dcrs": dcrs,
                "cli": cli,
                'lens': None,
            })
        else:
            samples.append({
                "reference": reference_text,
                "generated_caption": candidate_text,
                "BLEU": bleu_scores,
                "meteor": meteor,
                "rouge": rouge_scores,
                "cosine_similarity": cosine_scores
            })

    # Save results
    with open("evaluation_results_general.json", "w", encoding="utf-8") as out_f:
        json.dump(samples, out_f, indent=2)
    print("Evaluation complete. Results saved to evaluation_results.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate medical text generation outputs.")
    parser.add_argument('--prediction_file', type=str, required=True,default= 'BioLaySumm2025-eLife_result.json', help='Path to the predictions JSON file.')
    parser.add_argument('--groundtruth_file', type=str, default= 'BioLaySumm2025-eLife_result.json',required=True, help='Path to the ground truth JSON file.')
    parser.add_argument('--task_name',  type=str,  default= 'Lay_Summarisation', required=True, help='The name of the task.') #"Lay_Summarisation" "Radiology_Report_Generation"
    args = parser.parse_args()

    evaluate(args.prediction_file, args.groundtruth_file, args.task_name)