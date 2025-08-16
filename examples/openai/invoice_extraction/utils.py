from io import BytesIO
from collections import defaultdict
import base64
import json
import mlflow
import openai
import pandas as pd
from tqdm import tqdm


def pil_to_base64(pil_image):
    buffer = BytesIO()
    pil_image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# def process_dataset(model_name, dataset, client, system_prompt):
#     """Run inference and collect responses and ground truths."""
#     response_list = []
#     ground_truth_list = []
#     for data in tqdm(dataset):
#         image = data['image']
#         ground_truth = json.loads(data['ground_truth'])['gt_parse']
#         image_base64 = pil_to_base64(image)
#         response = run_openai_inference(model_name=model_name, client=client, image_base64=image_base64, system_prompt=system_prompt)
#         response_list.append(response)
#         ground_truth_list.append(ground_truth)
#     return response_list, ground_truth_list



def generate_urls(dataset):
    url_list = []
    ground_truth_list = []
    for data in tqdm(dataset):
        image = data['image']
        ground_truth = json.loads(data['ground_truth'])['gt_parse']
        image_base64 = pil_to_base64(image)
        url_list.append(image_base64)
        ground_truth_list.append(ground_truth)
    return url_list, ground_truth_list


# # @mlflow.trace(span_type="LLM")
# def run_openai_inference(model_name, client, image_base64, system_prompt):
#     """Send image and prompt to OpenAI model and return response."""
#     response = client.responses.create(
#         model=model_name,
#         reasoning={
#             "effort": "medium",
#         },
#         input=[
#             {
#                 "role": "user",
#                 "content": [
#                     { "type": "input_text", "text": system_prompt },
#                     {
#                         "type": "input_image",
#                         "image_url": f"data:image/jpeg;base64,{image_base64}",
#                         # "detail": "low"
#                     },
#                 ],
#             }
#         ],
#     )
#     return response

def flatten_json(y, prefix=''):
    """Flatten nested JSON into dot notation keys."""
    out = {}
    def flatten(x, name=''):
        if isinstance(x, dict):
            for a in x:
                flatten(x[a], f"{name}{a}.")
        elif isinstance(x, list):
            for i, a in enumerate(x):
                flatten(a, f"{name}{i}.")
        else:
            out[name[:-1]] = x
    flatten(y, prefix)
    return out


def apply_postprocessing(data):
    # if the key contains price replace comma by . in the price columns
    for key in data.keys():
        if "price" in key.lower():
            data[key] = data[key].replace(",", ".")
    return data

def calculate_invoice_accuracies(ground_truth_list, response_list):
    """Calculate per-invoice accuracy and return a DataFrame."""
    invoice_metrics = []
    for i in range(len(ground_truth_list)):
        gt = ground_truth_list[i]
        pred = response_list[i]
        # If response is an OpenAI object, parse output_text
        if hasattr(pred, "output_text"):
            pred = json.loads(pred.output_text)
        gt_flat = flatten_json(gt)
        gt_flat = apply_postprocessing(gt_flat)
        pred_flat = flatten_json(pred)
        pred_flat = apply_postprocessing(pred_flat)
        total_keys = len(gt_flat)
        matched_keys = sum(
            str(gt_flat[k]).strip() == str(pred_flat.get(k, "")).strip()
            for k in gt_flat
        )
        accuracy = matched_keys / total_keys if total_keys > 0 else 0.0
        invoice_metrics.append({
            "invoice_no": i,
            "total_keys": total_keys,
            "matched_keys": matched_keys,
            "accuracy": accuracy
        })
    invoice_metrics_df = pd.DataFrame(invoice_metrics)
    return invoice_metrics_df


def key_level_metrics(gt_list, pred_list):
    key_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    for gt_json, pred_json in zip(gt_list, pred_list):
        gt_flat = flatten_json(gt_json)
        gt_flat = apply_postprocessing(gt_flat)
        pred_flat = flatten_json(pred_json)
        pred_flat = apply_postprocessing(pred_flat)
        gt_keys = set(gt_flat.keys())
        pred_keys = set(pred_flat.keys())
        all_keys = gt_keys | pred_keys
        for k in all_keys:
            gt_val = gt_flat.get(k)
            pred_val = pred_flat.get(k)
            if gt_val is not None and pred_val is not None:
                if gt_val == pred_val:
                    key_stats[k]['tp'] += 1
                else:
                    print(f"False positive for key '{k}': GT='{gt_val}', Pred='{pred_val}'")
                    key_stats[k]['fp'] += 1  # Value present but incorrect
            elif gt_val is not None and pred_val is None:
                print(f"False negative for key '{k}': GT='{gt_val}', Pred=None")
                key_stats[k]['fn'] += 1   # Value missing in prediction
            elif gt_val is None and pred_val is not None:
                print(f"False positive for key '{k}': GT=None, Pred='{pred_val}'")
                key_stats[k]['fp'] += 1   # Extra key in prediction
    
    # Calculate metrics per key
    metrics = {}
    for k, stats in key_stats.items():
        tp, fp, fn = stats['tp'], stats['fp'], stats['fn']
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        metrics[k] = {'precision': precision, 'recall': recall, 'f1': f1, 'tp': tp, 'fp': fp, 'fn': fn}
    return metrics

