import os
from typing import Any, Dict, Optional, List

import torch
import torch.distributed as dist

from backbone.shared.data import ImageFinder
from config.backbone.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from backbone.shared.multimodal.conversation import SeparatorStyle, conv_templates
from backbone.shared.multimodal.data_processor import process_images, tokenizer_image_token
from backbone.shared.multimodal.generation_utils import KeywordsStoppingCriteria
from backbone.shared.runtime_logging import disable_torch_init

from .inference_engine import (
    BaseInferenceAdapter,
    InferenceContext,
    decode_new_tokens,
    default_generate,
    read_image,
)


def _vision_input_device_dtype(context: InferenceContext) -> tuple[torch.device, torch.dtype]:
    """Align image tensors with the loaded MLLM vision tower (device + dtype)."""
    vt = context.model.get_vision_tower() if hasattr(context.model, "get_vision_tower") else None
    if vt is not None:
        try:
            p = next(vt.parameters())
            if not p.is_meta:
                return p.device, p.dtype
        except StopIteration:
            pass
    from core.load_model import _resolve_model_device

    dev = _resolve_model_device(context.model)
    llm_dtype = getattr(context.model, "dtype", None)
    if llm_dtype is None:
        llm_dtype = torch.float16
    return dev, llm_dtype


class DefaultTaskAdapter(BaseInferenceAdapter):
    def __init__(self, recursive_image_search: bool = True, auto_mmtag_for_plain: bool = False) -> None:
        self.recursive_image_search = recursive_image_search
        self.auto_mmtag_for_plain = auto_mmtag_for_plain
        self.image_finder: Optional[ImageFinder] = None

    def on_model_ready(self, context: InferenceContext) -> None:
        disable_torch_init()
        if self.recursive_image_search and getattr(context.args, "image_folder", None):
            self.image_finder = ImageFinder(context.args.image_folder)

        if self.auto_mmtag_for_plain:
            model_name = context.model_name
            conv_mode = context.args.conv_mode
            if "plain" in model_name and "finetune" not in model_name.lower() and "mmtag" not in conv_mode:
                context.args.conv_mode = conv_mode + "_mmtag"
                print(
                    f"It seems that this is a plain model, but it is not using a mmtag prompt, "
                    f"auto switching to {context.args.conv_mode}."
                )

    def _build_question_with_image_token(self, question: str, model_config: Any) -> str:
        if getattr(model_config, "mm_use_im_start_end", False):
            return DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + question
        return DEFAULT_IMAGE_TOKEN + "\n" + question

    def _resolve_image_path(self, image_file: str, image_folder: str) -> str:
        if self.image_finder is not None:
            return self.image_finder.find(image_file)
        return os.path.join(image_folder, image_file)

    def infer_one(self, sample: Dict[str, Any], context: InferenceContext) -> Dict[str, str]:
        return self.infer_batch([sample], context)[0]

    def infer_batch(self, samples: List[Dict[str, Any]], context: InferenceContext) -> List[Dict[str, str]]:
        prompts = []
        input_ids_list = []
        images_list = []

        for sample in samples:
            question = sample["text"]
            qs = self._build_question_with_image_token(
                question, context.model.config)

            conv = conv_templates[context.args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            prompts.append(question)

            image_file = sample.get("image", None)
            if image_file:
                image_path = self._resolve_image_path(
                    image_file, context.args.image_folder)
                image = read_image(image_path)
                image_tensor = process_images(
                    [image], context.image_processor, context.model.config)[0]
                images_list.append(image_tensor)

            input_id = tokenizer_image_token(
                prompt, context.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            input_ids_list.append(input_id)

        # LLaVA `prepare_inputs_labels_for_multimodal` treats left-pad tokens as real
        # tokens. pad_token often equals eos, so batched ArxivQA (high length variance)
        # collapses to "." / "Ъ". Generate independently when lengths differ.
        if len(samples) > 1:
            lengths = [int(ids.numel()) for ids in input_ids_list]
            if max(lengths) != min(lengths):
                return [self.infer_batch([sample], context)[0] for sample in samples]

        pad_token_id = context.tokenizer.pad_token_id if context.tokenizer.pad_token_id is not None else context.tokenizer.eos_token_id
        max_len = max(len(ids) for ids in input_ids_list)

        padded_input_ids = []
        attention_mask = []
        for ids in input_ids_list:
            pad_len = max_len - len(ids)
            padded = torch.cat(
                [torch.full((pad_len,), pad_token_id, dtype=torch.long), ids])
            mask = torch.cat([torch.zeros(pad_len, dtype=torch.long),
                             torch.ones(len(ids), dtype=torch.long)])
            padded_input_ids.append(padded)
            attention_mask.append(mask)

        input_ids = torch.stack(padded_input_ids).to(
            device="cuda", non_blocking=True)
        attention_mask = torch.stack(attention_mask).to(
            device="cuda", non_blocking=True)

        if len(images_list) > 0:
            img_device, img_dtype = _vision_input_device_dtype(context)
            if all(img.shape == images_list[0].shape for img in images_list):
                images = torch.stack(images_list).to(
                    dtype=img_dtype, device=img_device, non_blocking=True)
            else:
                images = [
                    img.to(dtype=img_dtype, device=img_device, non_blocking=True)
                    for img in images_list
                ]
        else:
            images = None

        output_ids = default_generate(
            context.model, input_ids, images, context.args, attention_mask=attention_mask
        )

        outputs = []
        for i in range(len(samples)):
            out_text = context.tokenizer.decode(
                output_ids[i, max_len:], skip_special_tokens=True).strip()
            outputs.append({
                "prompt": prompts[i],
                "text": out_text,
            })

        return outputs


class ScienceQATaskAdapter(BaseInferenceAdapter):
    """
    ScienceQA task adapter.

    Anchors are loaded by the base stack / load_model_for_inference; this class only implements inference.
    """

    def on_model_ready(self, context: InferenceContext) -> None:
        """Called when the model is ready."""
        disable_torch_init()

        if hasattr(context.model, 'initialize_text_modules'):
            context.model.initialize_text_modules(context.args)
            print("Text tower modules initialized")

        self._verify_anchors_loaded(context.model)

    def _verify_anchors_loaded(self, model: Any) -> None:
        """Check anchor tensors exist."""
        missing = []
        for attr in ['image_anchors', 'text_anchors', 'image_boundary', 'text_boundary']:
            if not hasattr(model, attr):
                missing.append(attr)

        if missing:
            print(f"Warning: Missing anchors: {missing}")
        else:
            num_tasks = len(model.image_anchors) if hasattr(
                model, 'image_anchors') else 0
            print(f"Anchors loaded for {num_tasks} tasks")

    def infer_one(self, sample: Dict[str, Any], context: InferenceContext) -> Dict[str, str]:
        return self.infer_batch([sample], context)[0]

    def infer_batch(self, samples: List[Dict[str, Any]], context: InferenceContext) -> List[Dict[str, str]]:
        if getattr(context.args, "answer_prompter", False):
            # Fallback to serial for answer_prompter logic due to its interactive step
            outputs = []
            for sample in samples:
                outputs.append(self._infer_one_serial(sample, context))
            return outputs

        prompts = []
        input_ids_list = []
        images_list = []

        for sample in samples:
            question = sample["text"]
            qs = question.replace("<image>", "").strip()
            cur_prompt = qs

            if "image" in sample:
                image = read_image(os.path.join(
                    context.args.image_folder, sample["image"]))
                image_tensor = context.image_processor.preprocess(
                    image, return_tensors="pt")["pixel_values"][0]
                images_list.append(image_tensor)

                if getattr(context.model.config, "mm_use_im_start_end", False):
                    qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + \
                        DEFAULT_IM_END_TOKEN + "\n" + qs
                else:
                    qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
                cur_prompt = "<image>\n" + cur_prompt
            else:
                pass

            if getattr(context.args, "single_pred_prompt", False):
                suffix = "Answer with the option's letter from the given choices directly."
                qs = qs + "\n" + suffix
                cur_prompt = cur_prompt + "\n" + suffix

            conv = conv_templates[context.args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            prompts.append(cur_prompt)

            input_id = tokenizer_image_token(
                prompt, context.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            input_ids_list.append(input_id)

        if len(samples) > 1:
            lengths = [int(ids.numel()) for ids in input_ids_list]
            if max(lengths) != min(lengths):
                return [self.infer_batch([sample], context)[0] for sample in samples]

        pad_token_id = context.tokenizer.pad_token_id if context.tokenizer.pad_token_id is not None else context.tokenizer.eos_token_id
        max_len = max(len(ids) for ids in input_ids_list)
        padded_input_ids = []
        attention_mask = []
        for ids in input_ids_list:
            pad_len = max_len - len(ids)
            padded = torch.cat(
                [torch.full((pad_len,), pad_token_id, dtype=torch.long), ids])
            mask = torch.cat([torch.zeros(pad_len, dtype=torch.long),
                             torch.ones(len(ids), dtype=torch.long)])
            padded_input_ids.append(padded)
            attention_mask.append(mask)

        input_ids = torch.stack(padded_input_ids).cuda()
        attention_mask = torch.stack(attention_mask).cuda()

        if len(images_list) > 0:
            if all(img.shape == images_list[0].shape for img in images_list):
                images = torch.stack(images_list).half().cuda()
            else:
                images = [img.half().cuda() for img in images_list]
        else:
            images = None

        conv = conv_templates[context.args.conv_mode].copy()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

        output_ids = default_generate(
            context.model,
            input_ids,
            images,
            context.args,
            attention_mask=attention_mask,
            max_new_tokens=getattr(context.args, "max_new_tokens", 1024),
        )

        outputs = []
        for i in range(len(samples)):
            out_text = context.tokenizer.decode(
                output_ids[i, max_len:], skip_special_tokens=True).strip()
            if out_text.endswith(stop_str):
                out_text = out_text[:-len(stop_str)]
            out_text = out_text.strip()

            outputs.append({
                "prompt": prompts[i],
                "text": out_text,
            })

        return outputs

    def _infer_one_serial(self, sample: Dict[str, Any], context: InferenceContext) -> Dict[str, str]:
        question = sample["text"]
        qs = question.replace("<image>", "").strip()
        cur_prompt = qs

        # Images
        images = None
        if "image" in sample:
            image = read_image(os.path.join(
                context.args.image_folder, sample["image"]))
            image_tensor = context.image_processor.preprocess(
                image, return_tensors="pt")["pixel_values"][0]
            images = image_tensor.unsqueeze(0).half().cuda()

            if getattr(context.model.config, "mm_use_im_start_end", False):
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + \
                    DEFAULT_IM_END_TOKEN + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
            cur_prompt = "<image>\n" + cur_prompt

        # Optional answer-choice suffix
        if context.args.single_pred_prompt:
            suffix = "Answer with the option's letter from the given choices directly."
            qs = qs + "\n" + suffix
            cur_prompt = cur_prompt + "\n" + suffix

        # Build conversation
        conv = conv_templates[context.args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # Tokenize
        input_ids = tokenizer_image_token(
            prompt, context.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        input_ids = input_ids.unsqueeze(0).cuda()

        # Stop string
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = [KeywordsStoppingCriteria(
            keywords, context.tokenizer, input_ids)] if conv.version == "v0" else None

        # Generate
        output_ids = default_generate(
            context.model,
            input_ids,
            images,
            context.args,
            max_new_tokens=getattr(context.args, "max_new_tokens", 1024),
            stopping_criteria=stopping_criteria,
        )
        outputs = decode_new_tokens(
            context.tokenizer, output_ids, input_ids).strip()

        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        # Optional second-stage answer extraction
        if context.args.answer_prompter:
            outputs_reasoning = outputs
            prompt_with_answer = prompt + outputs_reasoning + " ###\nANSWER:"
            answer_input_ids = tokenizer_image_token(
                prompt_with_answer,
                context.tokenizer,
                IMAGE_TOKEN_INDEX,
                return_tensors="pt",
            ).unsqueeze(0).cuda()

            answer_output_ids = default_generate(
                context.model,
                answer_input_ids,
                images,
                context.args,
                max_new_tokens=64,
                stopping_criteria=stopping_criteria,
            )
            answer_text = decode_new_tokens(
                context.tokenizer, answer_output_ids, answer_input_ids).strip()
            if answer_text.endswith(stop_str):
                answer_text = answer_text[:-len(stop_str)]
            answer_text = answer_text.strip()
            outputs = outputs_reasoning + "\n The answer is " + answer_text

        return {
            "prompt": cur_prompt,
            "text": outputs,
        }
