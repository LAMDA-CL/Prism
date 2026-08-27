import os
import argparse
import json
import re
from multiprocessing import Pool, cpu_count
from collections import Counter
from collections import defaultdict

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation-file', type=str, default='./playground/Instructions_slim/VizWiz/val_new.json')
    parser.add_argument('--result-file', type=str, default='./results/CoIN_slim_new/VizWiz/Zero_shot/merge.jsonl')
    parser.add_argument('--output-dir', type=str)
    return parser.parse_args()

def _coco_file_to_id(annotation_file: str) -> dict:
    """Map annotation ``file_name`` / stem / trailing id to coco ``image_id``."""
    with open(annotation_file, "r", encoding="utf-8") as f:
        gt = json.load(f)
    mapping = {}
    for im in gt.get("images", []):
        iid = int(im["id"])
        fn = str(im.get("file_name") or "")
        base = os.path.basename(fn)
        stem, _ = os.path.splitext(base)
        mapping[fn] = iid
        mapping[base] = iid
        mapping[stem] = iid
        if "_" in stem:
            mapping[stem.split("_")[-1]] = iid
    return mapping


def _result_image_id(result: dict, file_to_id: dict, fallback: int) -> int:
    qid = str(result.get("question_id", "")).strip()
    if qid:
        for key in (qid, qid + ".jpg", qid + ".png", qid + ".jpeg"):
            if key in file_to_id:
                return int(file_to_id[key])
        if qid.isdigit() and qid in file_to_id:
            return int(file_to_id[qid])
    return int(fallback)


def create_coco_type(annotation_file, result_file, output_dir):
    results = [json.loads(line) for line in open(result_file)]

    pred_list = []
    total = len(results)
    right = 0
    coco_results = []
    file_to_id = _coco_file_to_id(annotation_file) if annotation_file else {}
    image_id = 1
    for result in results:
        pred = result['text']
        mapped = _result_image_id(result, file_to_id, image_id)
        coco_results.append({
            "image_id": int(mapped),
            "caption": pred
        })
        image_id += 1
    output_file = 'pred_coco_type.json'
    output_path = os.path.join(output_dir, output_file)
    with open(output_path, 'w') as f_out:
        json.dump(coco_results, f_out, indent=4)
    return output_path, total

def load_json(file_path):
    """Load a JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def merge_captions(pred_file, val_file, output_file):

    pred_data = load_json(pred_file)
    val_data = load_json(val_file)


    val_dict = defaultdict(list)
    for item in val_data['annotations']:
        val_dict[item['image_id']].append(item['caption'])


    merged_data = []
    for pred_item in pred_data:
        image_id = pred_item['image_id']
        pred_caption = pred_item['caption']


        gt_captions = val_dict.get(image_id, [])


        merged_data.append({
            "pred": pred_caption,
            "ground_truth": gt_captions
        })


    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, indent=4, ensure_ascii=False)

def eval_single(output_file, annotation_file, total, args):
    coco = COCO(annotation_file)  # Ground truth JSON file
    coco_res = coco.loadRes(output_file)  # Prediction JSON file

    coco_eval = COCOEvalCap(coco, coco_res)

    coco_eval.evaluate()

    metrics_to_print = ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "METEOR", "ROUGE_L", "CIDEr"]
    results = []
    for metric, score in coco_eval.eval.items():
        if metric in metrics_to_print:
            score_percentage = score * 100.
            print(f"{metric}: {score_percentage:.2f}")
            results.append(score_percentage)
        

    print('Samples: {}\nAverage: {:.2f}%\n'.format(total, sum(results) / len(results)))

    if args.output_dir is not None:
        output_file = os.path.join(args.output_dir, 'Result.text')
        with open(output_file, 'w') as f:
            f.write('Samples: {}\nBleu_1: {:.2f}\nBleu_2: {:.2f}\nBleu_3: {:.2f}\nBleu_4: {:.2f}\nMETEOR: {:.2f}\nROUGE_L: {:.2f}\nCIDEr: {:.2f}\nAverage: {:.2f}\n'.format(
                total, results[0], results[1], results[2], results[3], results[4], results[5], results[6], sum(results) / len(results)))
    

def process_batch(api_key, batch):
    """
    Score one batch and return per-sample scores from the LLM judge API.
    """

    client = OpenAI(api_key=api_key, base_url="https://platform.llmprovider.ai/v1")

    message = (
        "Below are the model's predictions and the ground truth answers for a task. "
        "For each case, provide a semantic similarity score between 0 and 10 in the format 'Score: X', "
        "where X is your score. Always use the format 'Score:' and do not explain anything."
        "\n\nResults:\n" +
        "\n".join([f"{i+1}. Pred: {item['pred']}, Ground Truth: {item['ground_truth']}" for i, item in enumerate(batch)])
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "You are an AI assistant evaluating a model's prediction quality"}, {"role": "user", "content": message}],
        stream=False
    )

    evaluation_text = response.choices[0].message.content


    scores = []
    for line in evaluation_text.splitlines():
        score = float(line.split(":")[1].strip())
        scores.append(score)
    return scores

def deepseek_chat_final(api_key, path, batch_size=10):
    """
    Multi-process LLM judge scoring; returns aggregate accuracy-style stats.
    """

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)


    batches = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
    num_batches = len(batches)

    print(f"Total data: {len(data)}, Total batches: {num_batches}, Batch size: {batch_size}")


    total_score = 0
    total_samples = 0
    with Pool(cpu_count()) as pool:
        results = pool.starmap(
            process_batch, [(api_key, batch) for batch in batches]
        )


        for batch_scores in results:
            total_score += sum(batch_scores)
            total_samples += len(batch_scores)


    overall_average_score = total_score / total_samples if total_samples > 0 else 0
    return overall_average_score
    



if __name__ == "__main__":
    args = get_args()

    if args.result_file is not None:
        output_file, total = create_coco_type(args.annotation_file, args.result_file, args.output_dir)
        ans_gt_file = os.path.join(args.output_dir, 'ans_gt.json')
        merge_captions(output_file, args.annotation_file, ans_gt_file)
        eval_single(output_file, args.annotation_file, total)

        # api_key = "sk-GdmqmU6fFWv5N0HlvYluLFzIbXIPNg3MHzPGeeV247092807Ba2e4487B9D5796cA3Be7dD4"

        # batch_size = 8 
        # overall_accuracy = deepseek_chat_final(api_key, ans_gt_file, batch_size=batch_size)
        # print(f"Overall Accuracy: {overall_accuracy*10:.2f}")
        # if args.output_dir is not None:
        #     output_file = os.path.join(args.output_dir, 'Result_api.text')
        #     with open(output_file, 'w') as f:
        #         f.write('Accuracy: {:.2f}%\n'.format(overall_accuracy*10))