import argparse
import torch
from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)
import tqdm
from PIL import Image

import requests
from PIL import Image
from io import BytesIO
import re
# from transformers import PreTrainedTokenizer
def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out

def unwrap_model(model):
    """
    Unwrap a model from a DataParallel or DistributedDataParallel wrapper.
    """
    if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
        return model.module
    else:
        return model
def uncache_media(model):
    if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
        model.module.cached_past_key_values = None
        model.module.cached_logits = None
        if hasattr(model.modulel, "cache_media_state"):
            model.module.cache_media_state = None
        torch.cuda.empty_cache()
    else:
        model.cached_past_key_values = None
        model.cached_logits = None
        if hasattr(model, "cache_media_state"):
            model.cache_media_state = None
        torch.cuda.empty_cache()

def get_predicted_classnames(logprobs, k, class_id_to_name):
    """
    Args:
        - logprobs shape (B, Y) containing logprobs for each classname
        - k: number for top-k
        - class_id_to_name: dict mapping class index to classname

    Returns:
        - top-k predicted classnames shape (B, k) type str
        - top-k logprobs shape (B, k) type float
    """
    # convert indices to classnames
    _, predictions = torch.topk(logprobs, k=k, dim=1)  # shape (B, k)
    predicted_classnames = [
        [class_id_to_name[ix] for ix in item] for item in predictions.tolist()
    ]
    predicted_logprobs = torch.gather(logprobs, 1, predictions)
    return predicted_classnames, predicted_logprobs

def get_rank_classifications(model, ctx_input_ids, batch_images, ctx_attention_mask, precomputed, all_class_names,  device, tokenizer, normalize_length=True, use_cache=True):
    """
    Returns a (B, |all_class_names|) tensor containing the logprobs for each class name.
    """
    if use_cache:
        precomputed_logits = precomputed.logits
        precomputed_pkvs = precomputed.past_key_values
        batch_size = precomputed_logits.size()[0]
    else:
        precomputed_pkvs = None

    #循环遍历类名：
    # Loop through class names and get log-likelihoods
    # Note: if all classnames are one token, this code is redundant, since we could
    # get all logits after one pass. However, if there are multi-token classnames,
    # we need to loop through each classname separately.
    overall_probs = []
    # i=0
    for class_name in all_class_names:
        # print('i=',i)
        # Tokenize only the class name
        classname_tokens = tokenizer(class_name, add_special_tokens=False, return_tensors="pt"
)["input_ids"].to(device)
        classname_tokens = torch.tensor(classname_tokens, dtype=torch.long)

        assert classname_tokens.ndim == 2
        # print('classname_tokens.',classname_tokens.size())
        classname_tokens = classname_tokens.repeat(batch_size, 1)
        num_tokens_in_classname = classname_tokens.shape[1]

        # Concatenate the class name tokens
        if not use_cache:
            _lang_x = torch.cat([ctx_input_ids, classname_tokens], dim=1)
            _attention_mask = torch.cat(
                [
                    ctx_attention_mask,
                    torch.ones_like(classname_tokens).bool(),
                ],
                dim=1,
            )
            _vision_x = batch_images
        else:
            _lang_x = classname_tokens
            _attention_mask = None
            _vision_x = None

        # Call forward to get the logits
        if isinstance(model, torch.nn.DataParallel):
            with torch.no_grad():
                outputs = model.module(
                    _lang_x,
                    images=_vision_x,
                    attention_mask=_attention_mask,
                    past_key_values=precomputed_pkvs,
                )
        else:
            with torch.no_grad():
                outputs = model(
                    _lang_x,
                    images=_vision_x,
                    attention_mask=_attention_mask,
                    past_key_values=precomputed_pkvs,
                )

        # Get the logits of the classname
        # logits shape is either (B, num_tokens_in_classname, vocab_len) with use_cache
        # or (B, len(_lang_x), vocab_len) without use_cache
        # remember that the logits at index t on dim 1 correspond to predictions for the t+1st token
        logits = outputs.logits
        if use_cache:
            logits = torch.cat([precomputed_logits, logits], dim=1)

        logprobs = torch.log_softmax(logits, dim=-1)
        gen_probs = logprobs[
            :, -num_tokens_in_classname - 1 : -1, :
        ]  # (B, num_tokens_in_classname, vocab_len)
        gen_probs = torch.gather(
            gen_probs, 2, classname_tokens[:, :, None]
        ).squeeze(-1)

        # Aggregate over tokens in the classname
        if normalize_length:
            class_prob = torch.mean(gen_probs, dim=1)
        else:
            class_prob = torch.sum(gen_probs, dim=1)
        overall_probs.append(class_prob)  # (B, 1)

        uncache_media(model)
        # i+=1
    overall_probs = torch.vstack(overall_probs).T.cpu()  # shape (B, num_classes)
    return overall_probs

def eval_model(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )

    qs = args.query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image_files = image_parser(args)
    images = load_images(image_files)
    image_sizes = [x.size for x in images]
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    print(outputs)
    return outputs


def eval_model_distrib(args, dataset, class_names):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )

    qs = args.query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN

    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs    ###'<image>\n+qs'

    if args.num_gpus > 1:
        model = DataParallel(model, device_ids=list(range(args.num_gpus)))
        device = torch.device(f'cuda:{model.device_ids[0]}')  # Use the primary device of DataParallel
        # print('device:', device)
    else:
        device = torch.device("cuda")
    model.to(device)


    if isinstance(model, torch.nn.DataParallel):
        model_config = model.module.config
    else:
        model_config = model.config

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    # Create the dataset
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    results = []
    conv = conv_templates[args.conv_mode].copy()
    for batch_idx, batch in enumerate(dataloader):

        batch_queries, batch_image_paths, batch_classnames = batch
        # print('batch_image_paths', batch_image_paths)
        # print('batch_classnames', batch_classnames)
        # Load and process images
        if args.few_shot == 0:
            images = load_images(batch_image_paths)
            image_sizes = [x.size for x in images]     ##list
            batch_images = process_images(
                images,
                image_processor,
                model_config
            ).to(device, dtype=torch.float16)

            batch_x = batch_images.size()[0]  ##[B,576,D]

            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = (
                tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                .unsqueeze(0).repeat(batch_x, 1)
            ).to(device)
            # print('input_ids', input_ids)
            conv.clear_message()
        else:
            batch_images = []
            image_sizes = []# Iterate over the batch (batch_image_paths and batch_classnames are 2D lists)
            batch_prompts = []
            max_length = 0
            #Transpose
            batch_classnames = list(zip(*batch_classnames))
            # 如果需要结果为真正的二维列表，而不是列表中包含元组，可进一步处理
            batch_classnames = [list(row) for row in batch_classnames]
            batch_image_paths = list(zip(*batch_image_paths))
            # 如果需要结果为真正的二维列表，而不是列表中包含元组，可进一步处理
            batch_image_paths = [list(row) for row in batch_image_paths]


            for image_paths, classnames in zip(batch_image_paths, batch_classnames):
                # Load images for a single data point (list of N paths)
                images = load_images(image_paths)  # Returns a list of PIL images or tensors
                # print('image_paths=', image_paths)
                # print('classnames', classnames)
                # Process each image individually
                single_data_images = process_images(images, image_processor, model_config) #[N, C, H, W]
                batch_images.append(single_data_images)
                # Store image sizes for further processing
                image_sizes.append([img.size for img in images])  # Store original sizes

                # support_str = ['A <image> liked picture is {}.'.format(cls) for cls in classnames[:-1]]
                support_str = ['A <image> liked photo is {}.'.format(cls) for cls in classnames[:-1]]
                # print('support_str', support_str)
                support_str = ' '.join(support_str) + 'Question:'
                # print('support_str:',support_str)

                conv.append_message(conv.roles[0], support_str + qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                # print('prompt=', prompt)
                tokenized_prompt = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                max_length = max(tokenized_prompt.size(0), max_length)  # 使用 F.pad 函数进行填充
                batch_prompts.append(tokenized_prompt)
                conv.clear_message()
                # print('batch_prompts', batch_prompts)

            # Combine all processed images into a batch tensor [Batch_size, N, C, H, W]
            batch_images = torch.stack(batch_images)
            # Convert to appropriate device and precision
            batch_images = batch_images.to(device, dtype=torch.float16).permute(1,0,2,3,4)
            # Tokenize the
            batch_prompts = [F.pad(pr, (0, (max_length-pr.size(0))), "constant", 0) for pr in batch_prompts]
            # print('batch_prompts', batch_prompts)
            input_ids = torch.stack(batch_prompts)   #[B,L]
            input_ids = input_ids.to(device)
        attention_mask = input_ids.ne(tokenizer.pad_token_id)
        if args.log_probs:
            with torch.inference_mode():
                if isinstance(model, torch.nn.DataParallel):
                    outputs = model.module(
                        input_ids,
                        images=batch_images,
                        attention_mask=attention_mask,
                        image_sizes=image_sizes,
                        use_cache=True,
                    )
                else:
                    outputs = model(
                        input_ids,
                        images=batch_images,
                        attention_mask=attention_mask,
                        image_sizes=image_sizes,
                        use_cache=True,
                    )
            logprobs = get_rank_classifications(model, input_ids, batch_images, attention_mask, outputs, class_names, device, tokenizer)
            class_id_to_name = dict(zip(range(len(class_names)), class_names))
            outputs, _ = get_predicted_classnames(
            logprobs,
            1,
            class_id_to_name,
        )
            # print('outputs = ', outputs)

        else:
            with torch.inference_mode():
                if isinstance(model, torch.nn.DataParallel):
                    output_ids = model.module.generate(
                        input_ids,
                        images=batch_images,
                        image_sizes=image_sizes,
                        do_sample=True if args.temperature > 0 else False,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        num_beams=args.num_beams,
                        max_new_tokens=args.max_new_tokens,
                        use_cache=True,
                    )
                else:
                    output_ids = model.generate(
                        input_ids,
                        images=batch_images,
                        do_sample=True if args.temperature > 0 else False,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        num_beams=args.num_beams,
                        max_new_tokens=args.max_new_tokens,
                        use_cache=True,
                    )

            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        # print(outputs)
        # Collect results

        for img_path, cls, query, output in zip(batch_image_paths, batch_classnames, batch_queries, outputs):
            if args.few_shot > 0:
                cls = cls[-1]
                img_path = img_path[-1]

            if type(output)==list:
                output = output[0]
            is_correct = cls.lower() in output.strip().lower()
            results.append({
                "image_path": img_path,
                "true_label": cls,
                "predicted_label": output,
                "is_correct": is_correct
            })

            # 可选：打印进度信息
            # if args.num_gpus > 1 and model.device_ids[0] == 0:
            print(f"Processed {img_path}: Predicted='{output}', True='{cls}', Correct={is_correct}")

        # if args.num_gpus > 1 and model.device_ids[0] == 0:
        print(f"Processed batch {batch_idx + 1}/{len(dataloader)}")

    # Calculate accuracy per class
    if args.few_shot>0:
        class_correct = {cls: 0 for cls in set(dataset.query_data[i]["classname"] for i in range(len(dataset)))}
    else:
        class_correct = {cls: 0 for cls in set(dataset.data[i]["classname"] for i in range(len(dataset)))}
    class_total = {cls: 0 for cls in class_correct}

    for result in results:
        classname = result["true_label"]
        class_total[classname] += 1
        if result["is_correct"]:
            class_correct[classname] += 1

    class_acc = {cls: [class_correct[cls] , class_total[cls]] for cls in class_correct}

    # if args.num_gpus > 1 and model.device_ids[0] == 0:
    print("Class-wise Accuracy:")
    for cls, it in class_acc.items():
        acc = it[0] / it[1] if it[1] > 0 else 0
        print(f"{cls}: {acc:.2%}")

    return results, class_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--few_shot", type=int, default=0)
    parser.add_argument("--log_probs", type=bool, default=True)


    args = parser.parse_args()

    eval_model(args)
