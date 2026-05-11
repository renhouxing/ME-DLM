import torch
import random
import logging
import datetime

import torch.distributed as dist
import torch.nn.functional as F

from transformers import Trainer, TrainerCallback, trainer

from edit_cpp import compute_labels_cpp

def seed_worker_patch(worker_id: int, num_workers: int, rank: int):
    trainer.set_seed(3407 + worker_id)

trainer.seed_worker = seed_worker_patch

logger = logging.getLogger()

class LoggerCallback(TrainerCallback):

    def on_train_begin(self, args, state, control, **kwargs):
        
        self.start_time = datetime.datetime.now()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_local_process_zero:
            return
        
        loss_msg = ' '.join(["%s: %.4f" % (k, v) for k, v in logs.items() if 'loss' in k or 'steps' in k])

        if loss_msg == '':
            return
        
        now = datetime.datetime.now()
        pass_time = now - self.start_time
        rest_time = pass_time * (state.max_steps - state.global_step) / state.global_step
        eta = now + rest_time

        pt_min = pass_time.seconds // 60
        pass_time = '%.2d:%.2d' % (pt_min // 60 + pass_time.days * 24, pt_min % 60)

        rt_min = rest_time.seconds // 60
        rest_time = '%.2d:%.2d' % (rt_min // 60 + rest_time.days * 24, rt_min % 60)

        logger.info(
            'step: %d epoch: %.4f %s lr: %.4g passed time: %s rest time: %s eta: %s',
            state.global_step, state.epoch, loss_msg, logs.get('learning_rate', 0),
            pass_time, rest_time, eta.strftime('%m/%d %H:%M')
        )

class EditCollator:

    def __init__(self, args, tokenizer):

        self.args = args
        self.tokenizer = tokenizer

        self._mask_id = tokenizer.mask_token_id
        self._del_id = tokenizer.convert_tokens_to_ids('<|del|>')
        self._im_end_id = tokenizer.convert_tokens_to_ids('<|im_end|>')
    
    def torch_uniform(self, low, high):
        x = (high - low) * random.random() + low
        return x

    def __call__(self, inputs):
        input_ids = torch.tensor(inputs[0]['input_ids']).long()
        token_labels = input_ids.clone().long()

        prompt_mask = []
        prompt_lens = []

        for i in range(len(inputs[0]['prompt_len'])):
            _prompt_len = inputs[0]['prompt_len'][i]
            begin = inputs[0]['cu_seq_len'][i]
            end = begin + _prompt_len + 1
            token_labels[begin:end] = -100

            prompt_mask.append(torch.ones((_prompt_len, ), dtype=torch.long))
            prompt_mask.append(torch.zeros((inputs[0]['cu_seq_len'][i + 1] - begin - _prompt_len, ), dtype=torch.long))

            prompt_lens.append(_prompt_len)
        
        rand_val = random.random()

        if rand_val < self.args.intermediate_ratio:
            mode = 'intermediate'
            t = self.torch_uniform(self.args.intermediate_min_t, self.args.intermediate_max_t)
        else:
            mode = 'mask'
            t = self.torch_uniform(self.args.min_t, self.args.max_t)
            
        mask_positions = torch.bernoulli(torch.full(input_ids.shape, t))
        mask_positions = mask_positions.masked_fill(token_labels == -100, 0).bool()

        input_ids = input_ids.masked_fill(mask_positions, self._mask_id)
        token_labels = token_labels.masked_fill(~mask_positions, -100)

        pad_num = self.args.max_len - input_ids.size(0)

        max_seq_len = max(inputs[0]['max_seq_len'], pad_num)
        cu_seq_len = torch.tensor(inputs[0]['cu_seq_len'] + [self.args.max_len]).int()

        input_ids = F.pad(input_ids, (0, pad_num), value=self.tokenizer.pad_token_id)
        token_labels = F.pad(token_labels, (0, pad_num), value=-100)
        prompt_mask = F.pad(torch.cat(prompt_mask), (0, pad_num), value=0)

        curr_token_labels = token_labels.unsqueeze(0)
        next_token_labels = curr_token_labels.roll(-1)
        
        return {
            "t": t,
            "mode": mode,
            "input_ids": input_ids.unsqueeze(0),
            "curr_token_labels": curr_token_labels,
            "next_token_labels": next_token_labels,
            "cu_seq_lens": cu_seq_len,
            "max_length": max_seq_len,
            "prompt_mask": prompt_mask,
            "prompt_lens": prompt_lens,
        }

class EditDiffusionLMTrainer(Trainer):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.add_callback(LoggerCallback)
        self.data_collator = EditCollator(self.args, self.processing_class)

        self._stored_metrics = {}

        self.torch_generator = torch.Generator()
        self.torch_generator.manual_seed(self.args.seed)

        self.del_id = self.processing_class.convert_tokens_to_ids('<|del|>')
        self.eos_id = self.processing_class.convert_tokens_to_ids('<|im_end|>')

        self.pad_id = self.processing_class.pad_token_id
        self.mask_id = self.processing_class.mask_token_id
    
    def reduce_tensor(self, tensor):

        world_size = dist.get_world_size()

        if world_size <= 1:
            return tensor.detach().nanmean().item()

        tensor = tensor.detach().nanmean()
        tensors = [torch.empty_like(tensor) for _ in range(world_size)]

        dist.all_gather(tensors, tensor)
        tensor = torch.stack(tensors, dim=0).nanmean()

        return tensor.item()
    
    def store_metrics(self, metrics):
        for key, value in metrics.items():
            if key not in self._stored_metrics:
                self._stored_metrics[key] = []
            self._stored_metrics[key].append(value)
    
    @torch.no_grad()
    def mask_diffusion(self, input_ids, cu_seq_lens, pred_per_step, model, max_prob_score=False):
        answer_mask = input_ids == self.mask_id

        max_seq_len = (cu_seq_lens[1:] - cu_seq_lens[:-1]).max().item()
        while answer_mask.any():
            outputs = model(
                input_ids=input_ids.unsqueeze(0), 
                cu_seq_lens_q=cu_seq_lens,
                cu_seq_lens_k=cu_seq_lens,
                max_length_q=max_seq_len,
                max_length_k=max_seq_len,
                is_causal=False,
                use_cache=False,
            )

            logit = outputs.logits[0]

            logit[:, self.del_id] = -torch.inf

            curr_token_ids = torch.argmax(logit, dim=-1)
            if max_prob_score:
                # logit[:, self.pad_id] = -torch.inf
                scores = torch.gather(torch.softmax(logit, dim=-1), dim=-1, index=curr_token_ids.unsqueeze(-1)).squeeze(-1)
            else:
                scores = torch.rand_like(input_ids, dtype=torch.float)
            scores[~answer_mask] = -torch.inf

            for i in range(1, cu_seq_lens.size(0)):
                total_num = answer_mask[cu_seq_lens[i - 1]:cu_seq_lens[i]].sum().item()
                pred_this_step = min(total_num, pred_per_step)

                if pred_this_step == 0:
                    continue

                pred_positions = torch.topk(
                    scores[cu_seq_lens[i - 1]:cu_seq_lens[i]], k=pred_this_step
                ).indices + cu_seq_lens[i - 1]

                answer_mask[pred_positions] = False
                input_ids[pred_positions] = curr_token_ids[pred_positions]
        
        return input_ids

    @torch.no_grad()
    def edit_diffusion(self, input_ids, prompt_mask, cu_seq_lens, model, diffusion_steps=32):

        for _ in range(diffusion_steps):

            max_seq_len = (cu_seq_lens[1:] - cu_seq_lens[:-1]).max().item()
            outputs = model(
                input_ids=input_ids.unsqueeze(0), 
                cu_seq_lens_q=cu_seq_lens,
                cu_seq_lens_k=cu_seq_lens,
                max_length_q=max_seq_len,
                max_length_k=max_seq_len,
                is_causal=False,
                use_cache=False,
            )

            curr_token_ids = torch.argmax(outputs.logits[0], dim=-1)
            next_token_ids = torch.argmax(outputs.next_token_logits[0], dim=-1).roll(1)

            # Setting Prompt
            curr_token_ids[prompt_mask.bool()] = input_ids[prompt_mask.bool()]
            next_token_ids[prompt_mask.bool()] = input_ids[prompt_mask.bool()]

            # DEL
            del_mask = torch.ne(curr_token_ids, self.del_id)
            curr_token_ids = curr_token_ids[del_mask]
            next_token_ids = next_token_ids[del_mask]
            prompt_mask = prompt_mask[del_mask]

            del_mask_int = torch.cat([
                torch.tensor([0], dtype=torch.int, device=del_mask.device),
                torch.cumsum(del_mask, dim=0)
            ], dim=0).int()
            cu_seq_lens = del_mask_int[cu_seq_lens]

            # INS
            save_mask = torch.ne(curr_token_ids, next_token_ids)

            # Avoid exceeding max length
            n = self.args.max_len - cu_seq_lens[-1].item()
            idx = save_mask.nonzero(as_tuple=True)[0]
            if idx.numel() > n:
                save_mask[idx[n:]] = False  

            all_mask = torch.stack([save_mask, torch.ones_like(curr_token_ids).bool()], dim=1).view(-1)

            all_prompt_mask = torch.stack([torch.zeros_like(curr_token_ids).bool(), prompt_mask], dim=1).view(-1)
            prompt_mask = all_prompt_mask[all_mask]

            all_mask_int = torch.cat([
                torch.tensor([0], dtype=torch.int, device=all_mask.device),
                torch.cumsum(all_mask, dim=0)
            ], dim=0).int()
            cu_seq_lens = all_mask_int[cu_seq_lens * 2]

            all_input_ids = torch.stack([next_token_ids, curr_token_ids], dim=1).view(-1)
            input_ids = all_input_ids[all_mask]

        return input_ids, cu_seq_lens

    @torch.no_grad()
    def construct_inputs(self, inputs, model):

        diffusion_steps = int((self.state.epoch - int(self.state.epoch)) * self.args.max_diffusion_steps)

        mask_pred_per_step = self.args.mask_diffusion_pred_per_step[torch.randint(
            low=0, high=len(self.args.mask_diffusion_pred_per_step), size=(1,), generator=self.torch_generator
        ).item()]

        input_ids = inputs['input_ids'].squeeze(0)
        labels = inputs['curr_token_labels'].squeeze(0)

        target_ids = input_ids.clone()
        mask = labels != -100
        target_ids[mask] = labels[mask]
        target_ids = target_ids.cpu().numpy().tolist()

        input_ids = self.mask_diffusion(
            input_ids, 
            inputs['cu_seq_lens'], 
            mask_pred_per_step,
            model,
        )

        eos_mask = input_ids != self.pad_id
        input_ids = input_ids[eos_mask]
        prompt_mask = inputs['prompt_mask'][eos_mask]

        cu_seq_lens = inputs['cu_seq_lens']
        eos_mask_int = torch.cat([
            torch.tensor([0], dtype=torch.int, device=eos_mask.device),
            torch.cumsum(eos_mask, dim=0)
        ], dim=0).int()
        cu_seq_lens = eos_mask_int[cu_seq_lens]
        
        if diffusion_steps > 0:
            input_ids, cu_seq_lens = self.edit_diffusion(
                input_ids, 
                prompt_mask, 
                cu_seq_lens, 
                model,
                diffusion_steps=diffusion_steps
            )

        input_ids_list = input_ids.cpu().numpy().tolist()

        curr_token_labels, next_token_labels = [], []
        for i in range(len(inputs['prompt_lens'])):
            _target_ids = target_ids[inputs['cu_seq_lens'][i]:inputs['cu_seq_lens'][i + 1]]
            _target_ids = [_id for _id in _target_ids if _id != self.pad_id]
            _input_ids = input_ids_list[cu_seq_lens[i]:cu_seq_lens[i + 1]]

            prompt_len = inputs['prompt_lens'][i]

            if len(_input_ids) <= prompt_len:
                curr_token_labels.extend([-100] * len(_input_ids))
                next_token_labels.extend([-100] * len(_input_ids))
                continue
            
            if self.state.global_step % 20 == 0 and i == 0:
                logger.info("Input\n%s", self.processing_class.decode(_input_ids, skip_special_tokens=False))
                logger.info("Target\n%s", self.processing_class.decode(_target_ids, skip_special_tokens=False))

            _curr_token_labels, _next_token_labels = compute_labels_cpp(
                _input_ids[prompt_len:-1], 
                _target_ids[prompt_len:-1], 
                del_id=self.del_id, 
                eos_id=self.eos_id,
                ignore_id=-100,
                pred_clean_token=True,
            )
            curr_token_labels.extend([-100] * prompt_len + _curr_token_labels + [self.eos_id])
            next_token_labels.extend([-100] * (prompt_len - 1) + _next_token_labels + [-100])

        curr_token_labels = torch.tensor(curr_token_labels, dtype=torch.long, device=input_ids.device)
        next_token_labels = torch.tensor(next_token_labels, dtype=torch.long, device=input_ids.device)

        max_seq_len = (cu_seq_lens[1:] - cu_seq_lens[:-1]).max().item()

        pad_num = self.args.max_len - input_ids.shape[0]

        if pad_num > 0:
            max_seq_len = max(max_seq_len, pad_num)
            cu_seq_lens = torch.cat([cu_seq_lens, torch.tensor([self.args.max_len], dtype=torch.int, device=cu_seq_lens.device)], dim=0)
            input_ids = F.pad(input_ids, (0, pad_num), value=self.pad_id)
            curr_token_labels = F.pad(curr_token_labels, (0, pad_num), value=-100)
            next_token_labels = F.pad(next_token_labels, (0, pad_num), value=-100)

        results = {
            "input_ids": input_ids.unsqueeze(0),
            "curr_token_labels": curr_token_labels.unsqueeze(0),
            "next_token_labels": next_token_labels.unsqueeze(0),
            "cu_seq_lens": cu_seq_lens,
            "max_length": max_seq_len,
        }
        
        return results, diffusion_steps

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        mode = inputs['mode']
        t = inputs['t']

        if mode == 'intermediate':
            inputs, diffusion_steps = self._prepare_inputs(self.construct_inputs(inputs, model))
            self.store_metrics(dict(intermediate_diffusion_steps=float(diffusion_steps)))

        if self.args.world_size > 1:
            dist.barrier()

        outputs = model(
            input_ids=inputs['input_ids'],
            curr_token_labels=inputs['curr_token_labels'],
            next_token_labels=inputs['next_token_labels'],
            cu_seq_lens_q=inputs['cu_seq_lens'],
            cu_seq_lens_k=inputs['cu_seq_lens'],
            max_length_q=inputs['max_length'],
            max_length_k=inputs['max_length'],
            use_cache=False, 
            is_causal=False,
            num_items_in_batch=num_items_in_batch
        )

        loss = outputs.loss + outputs.next_token_loss

        self.store_metrics({
            f"{mode}_time_steps": t,
            f"{mode}_loss": self.reduce_tensor(loss),
            f"{mode}_cur_token_loss": self.reduce_tensor(outputs.loss),
            f"{mode}_next_token_loss": self.reduce_tensor(outputs.next_token_loss),
        })

        return loss / t
    
    def log(self, logs, start_time=None):
        logs.pop('loss', None)
        for key, metrics in self._stored_metrics.items():
            if len(metrics) > 0:
                logs[key] = torch.tensor(metrics).mean().item()
        
        for key in self._stored_metrics:
            self._stored_metrics[key].clear()
        
        super().log(logs, start_time=None)