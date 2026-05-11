import math
import binpacking

class Processor:

    def __init__(self, max_len, pad_len, tokenizer):

        self.max_len = max_len
        self.pad_len = pad_len
        self.tokenizer = tokenizer

        self._bos_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        self._eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

        self.generation_prefix_len = len(self.tokenizer.encode(f"<|im_start|>assistant\n", add_special_tokens=False))

    def tokenize_sft(self, messages, response):

        input_ids = []
        for message in messages:
            string = f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
            input_ids += self.tokenizer.encode(string, add_special_tokens=False)
        
        prompt_len = len(input_ids) + self.generation_prefix_len
        input_ids += self.tokenizer.encode(f"<|im_start|>assistant\n{response}<|im_end|>", add_special_tokens=False)

        if self.pad_len is not None:
            pad_len = self.pad_len - (len(input_ids) - prompt_len) % self.pad_len
            input_ids += [self.tokenizer.pad_token_id] * pad_len

        return input_ids, prompt_len

    def tokenize_pretrain(self, text):

        input_ids = self.tokenizer.encode(text, add_special_tokens=False)

        input_ids = [self._bos_id] + input_ids + [self._eos_id]
        
        if self.pad_len is not None:
            pad_len = self.pad_len - len(input_ids) % self.pad_len
            input_ids += [self.tokenizer.pad_token_id] * pad_len

        return input_ids, 1
    
    def _group_texts(self, sample_input_ids, sample_prompt_lens):
        length = [(i, len(x)) for i, x in enumerate(sample_input_ids)]
        bins = binpacking.to_constant_volume(length, self.max_len, weight_pos=1)

        input_ids, prompt_len, cu_seq_len, max_seq_len = [], [], [], []

        for bin in bins:
            _input_ids, _prompt_len, _cu_seq_len, _max_seq_len = [], [], [0], 0
            for index in bin:
                _input_ids.extend(sample_input_ids[index[0]])
                _prompt_len.append(sample_prompt_lens[index[0]])
                _cu_seq_len.append(_cu_seq_len[-1] + len(sample_input_ids[index[0]]))
                _max_seq_len = max(_max_seq_len, len(sample_input_ids[index[0]]))
            
            assert len(_input_ids) <= self.max_len

            input_ids.append(_input_ids)
            prompt_len.append(_prompt_len)
            cu_seq_len.append(_cu_seq_len)
            max_seq_len.append(_max_seq_len)

        return {
            "input_ids": input_ids,
            "prompt_len": prompt_len,
            "cu_seq_len": cu_seq_len,
            "max_seq_len": max_seq_len,
        }

    def process_tokenize(self, examples):

        input_ids, prompt_lens = [], []
        if 'history' in examples and 'response' in examples:
            for history, response in zip(examples['history'], examples['response']):
                _input_ids, _prompt_len = self.tokenize_sft(history, response)

                if len(_input_ids) > self.max_len:
                    continue
                    
                input_ids.append(_input_ids)
                prompt_lens.append(_prompt_len)
        else:
            for text in examples['text']:
                _input_ids, _prompt_len = self.tokenize_pretrain(text)

                if len(_input_ids) > self.max_len:
                    continue
                    
                input_ids.append(_input_ids)
                prompt_lens.append(_prompt_len)
        
        return self._group_texts(input_ids, prompt_lens)
    