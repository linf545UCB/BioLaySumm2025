import os
import json
import argparse
import random
import numpy as np
from summac.model_summac import SummaCZS, SummaCConv



def evaluate(prediction_file, groundtruth_file, task_name):
    # Load predictions and references
    with open(prediction_file, 'r', encoding='utf-8') as pred_f:
        predictions = json.load(pred_f)
    with open(groundtruth_file, 'r', encoding='utf-8') as gt_f:
        groundtruths = json.load(gt_f)

    SummaCZS_model = SummaCZS(granularity="sentence", model_name="vitc", device="cuda") 

    samples = []
    for i, (pred, gt) in enumerate(zip(predictions, groundtruths)):
        if i % 100 == 0:
            print(f"Processing {i}")

        reference_text = gt["reference"]
        document_text = gt["document"]
        candidate_text = pred["generated_caption"]

        
        
        if task_name == 'Lay_Summarisation':

            # r = Readability(candidate_text) #LENS 这个指标要确认一下
            # lens = r.linsear_write()
        #     alignScoreCS.score(context=reference_text, claim=candidate_text)# AlignScore
            SummaC = SummaCZS_model.score([document_text], [candidate_text])["scores"][0] #SummaC
            
        # elif task_name == 'Radiology_Report_Generation':
        #     accuracy, accuracy_not_averaged, class_report, class_report_5 = f1chexbert(hyps=[candidate_text], refs=[reference_text])
        #     mean_reward, reward_list, hypothesis_annotation_lists, reference_annotation_lists = f1radgraph(hyps=[candidate_text], refs=[reference_text]) #f1radgraph
        #     #RadCliQ


        # print(reference_text)
        # print(candidate_text)
        # print(bleu_scores)
        # print(meteor)
        # print(rouge_scores)
        # print(cosine_scores)
        # input()
        if task_name == 'Lay_Summarisation':
            samples.append({
                "reference": reference_text,
                "generated_caption": candidate_text,
                "SummaC": SummaC,
            })

    # Save results
    with open("evaluation_results.json", "w", encoding="utf-8") as out_f:
        json.dump(samples, out_f, indent=2)
    print("Evaluation complete. Results saved to evaluation_results.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate medical text generation outputs.")
    parser.add_argument('--prediction_file', default= 'BioLaySumm2025-eLife_result.json', type=str, required=True, help='Path to the predictions JSON file.')
    parser.add_argument('--groundtruth_file',  default= 'BioLaySumm2025-eLife_result.json', type=str, required=True, help='Path to the ground truth JSON file.')
    parser.add_argument('--task_name',  default= 'Lay_Summarisation', type=str, required=True, help='The name of the task.') #"Lay_Summarisation" "Radiology_Report_Generation"
    args = parser.parse_args()

    evaluate(args.prediction_file, args.groundtruth_file, args.task_name)