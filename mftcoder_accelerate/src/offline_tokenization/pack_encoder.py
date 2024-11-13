from transformers import AutoTokenizer
from tokenizer import init_tokenizer


def load_tokenizer(model_path, tokenizer_type=None):
    """
    Load tokenizer from the given <model_path>
    """

    def load_tokenizer_manual(model_path, tokenizer_type):
        """
        Load tokenizer by the concrete Tokenizer class instead of AutoTokenizer
        """
        try:
            if tokenizer_type.lower() == "LlamaTokenizer".lower():
                return LlamaTokenizer.from_pretrained(model_path)

            raise Exception(f"Unsupported tokenizer type {tokenizer_type}")
        except:
            raise Exception(f"Unable to load tokenizer {tokenizer_type} from the given path: {model_path}")

    def load_tokenizer_auto(model_path):
        """
        Load tokenizer from the given path by HuggingFace AutoTokenizer
        """
        try:
            # tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)  # support CodeLlama
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            return tokenizer
        except:
            raise Exception(
                f'Unable to load tokenizer from the given path: {model_path} using auto mode.\nPlease specify the tokenizer type with the command argument "--tokenizer-type" and retry.'
            )

    # First, try to load tokenizer by huggingface AutoTokenizer, If fail, try another manual way
    try:
        return load_tokenizer_auto(model_path)
    except Exception as e:
        print(str(e))
        if tokenizer_type is not None:
            try:
                tokenizer = load_tokenizer_manual(model_path, tokenizer_type)
                return tokenizer
            except Exception as ee:
                raise ee


class PackPFTEncoder:
    """
    A sample of this format will be:
        <|role_start|>system<|role_end|> content of system_1
        <|role_start|>human<|role_end|> content of human_1
        <|role_start|>bot<|role_end|> content of bot_1
        <|endoftext|>
        <|role_start|>system<|role_end|> content of system_2
        <|role_start|>human<|role_end|> content of human_2
        <|role_start|>bot<|role_end|> content of bot_2
        <|endoftext|>
        <|role_start|>human<|role_end|> content of human_3
        <|role_start|>bot<|role_end|> content of bot_3
        <|endoftext|>
        ....
        <|role_start|>human<|role_end|> content of human_n
        <|role_start|>bot<|role_end|> content of bot_n
        <|endoftext|>
        <endoftext>
        <|pad|><|pad|>...<|pad|>

    system part is optional, i.e. '<|role_start|>system<|role_end|> content of system_i'
    """

    def __init__(self, seq_length, eod_token_id, pad_token_id, role_start_tag, role_end_tag, mode="pft"):
        self.mode = mode
        self.seq_length = seq_length
        self.eod_token_id = eod_token_id
        self.pad_token_id = pad_token_id
        self.role_start_tag = role_start_tag
        self.role_end_tag = role_end_tag

    def initializer(self, model_path, tokenizer_type=None):
        # Use Encoder class as a container for global data
        assert model_path is not None
        self.tokenizer = load_tokenizer(model_path, tokenizer_type)

    def encode(self, item):
        encode_res = {
            "input_ids": [],
        }

        item_len = sum([len(x["content"]) for x in item["chat_rounds"]])
        for token_res in self.tokenize_chat_prompt(item):
            for k, v in token_res.items():
                encode_res[k].append(v)
        return encode_res, item_len

    def tokenize_chat_prompt(self, item):
        # role_start_marker = self.tokenizer.encode(self.role_start_tag, add_special_tokens=False)
        # role_end_marker = self.tokenizer.encode(self.role_end_tag, add_special_tokens=False)
        end_marker = [self.eod_token_id]

        input_ids = []
        raw_input = ""
        # loss_mask = []
        for chat_round in item["chat_rounds"]:
            role = chat_round["role"].strip()
            # skip system prompt
            # if role == 'system':
            #    continue

            content = chat_round["content"]
            content = content if content.endswith("\n") else f"{content}\n"
            text = f"{self.role_start_tag}{role}{self.role_end_tag}{content}"
            chat_input_ids = self.tokenizer.encode(text, add_special_tokens=False)

            if role != "bot":
                chat_input_ids = chat_input_ids
            else:
                chat_input_ids = chat_input_ids + end_marker

            input_ids += chat_input_ids

        # if this sample's length is more than the specified max length, drop it
        # here, we don't add padding tokens for a single sample, however, we will append padding tokens for a combinated samaple
        if len(input_ids) > self.seq_length:
            yield {}
        else:
            yield {"input_ids": input_ids}

    def padding(self, key, data):
        assert len(data) <= self.seq_length, f"padding sequence: {len(data)} > {self.seq_length}"
        if key == "input_ids":
            return data + [self.pad_token_id] * (self.seq_length - len(data))

        if key == "loss_mask":
            return data + [0] * (self.seq_length - len(data))

        raise Exception("Should not reach here. There must be something wrong.")


class PackSFTEncoder:
    """
    A sample of this format will be:
        <|role_start|>system<|role_end|> content of system_1
        <|role_start|>human<|role_end|> content of human_1
        <|role_start|>bot<|role_end|> content of bot_1
        <|endoftext|>
        <|role_start|>system<|role_end|> content of system_2
        <|role_start|>human<|role_end|> content of human_2
        <|role_start|>bot<|role_end|> content of bot_2
        <|endoftext|>
        <|role_start|>human<|role_end|> content of human_3
        <|role_start|>bot<|role_end|> content of bot_3
        <|endoftext|>
        ....
        <|role_start|>human<|role_end|> content of human_n
        <|role_start|>bot<|role_end|> content of bot_n
        <|endoftext|>
        <endoftext>
        <|pad|><|pad|>...<|pad|>

    system part is optional, i.e. '<|role_start|>system<|role_end|> content of system_i'
    """

    def __init__(self, seq_length, eod_token, role_start_tag, role_end_tag, mode="sft"):
        self.mode = mode
        self.seq_length = seq_length
        self.eod_token = eod_token
        self.role_start_tag = role_start_tag
        self.role_end_tag = role_end_tag

    def initializer(self, model_path, tokenizer_type=None):
        # Use Encoder class as a container for global data
        assert model_path is not None
        self.tokenizer = load_tokenizer(
            model_path, tokenizer_type
        )  # AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    def encode(self, item):
        encode_res = {"input_ids": [], "raw_input": []}

        item_len = sum([len(x["content"]) for x in item["chat_rounds"]])
        for token_res in self.tokenize_chat_prompt(item):
            for k, v in token_res.items():
                encode_res[k].append(v)
        return encode_res, item_len

    def tokenize_chat_prompt(self, item):
        role_start_marker = self.tokenizer.encode(self.role_start_tag, add_special_tokens=False)
        role_end_marker = self.tokenizer.encode(self.role_end_tag, add_special_tokens=False)
        end_marker = [self.tokenizer.convert_tokens_to_ids(self.eod_token)]

        input_ids = []
        raw_input = ""
        # loss_mask = []
        for chat_round in item["chat_rounds"]:
            role = chat_round["role"]
            content = chat_round["content"]
            content = content if content.endswith("\n") else f"{content}\n"
            chat_input_ids = self.tokenizer.encode(content, add_special_tokens=False)
            role_input_ids = self.tokenizer.encode(role, add_special_tokens=False)
            role_raw_input = ""

            if role != "bot":
                # chat_loss_mask = [0] * len(role_start_marker) + [0] * len(role_input_ids) + [0] * len(role_end_marker) + [0] * len(chat_input_ids)
                chat_input_ids = role_start_marker + role_input_ids + role_end_marker + chat_input_ids
                role_raw_input = ROLE_START_MARKER + role + ROLE_END_MARKER + content
            elif role == "human":
                # chat_loss_mask = [0] * len(role_start_marker) + [0] * len(role_input_ids) + [0] * len(role_end_marker) + [1] * len(chat_input_ids) + [1] * len(end_marker)
                chat_input_ids = role_start_marker + role_input_ids + role_end_marker + chat_input_ids + end_marker
                role_raw_input = ROLE_START_MARKER + role + ROLE_END_MARKER + content + self.eod_token

            input_ids += chat_input_ids
            raw_input += role_raw_input
            # loss_mask += chat_loss_mask

        # assert len(input_ids) == len(loss_mask)

        # if this sample's length is more than the specified max length, drop it
        # here, we don't add padding tokens for a single sample, however, we will append padding tokens for a combinated samaple
        if len(input_ids) > self.seq_length:
            yield {}
        else:
            yield {
                "input_ids": input_ids,
                "raw_input": raw_input,
                # "loss_mask": loss_mask
            }

    def padding(self, key, data, pad_token_id):
        assert len(data) <= self.seq_length, f"padding sequence: {len(data)} > {self.seq_length}"
        if key == "input_ids":
            return data + [pad_token_id] * (self.seq_length - len(data))

        if key == "loss_mask":
            return data + [0] * (self.seq_length - len(data))

        raise Exception("Should not reach here. There must be something wrong.")


class PackSSTBinEncoder:
    """
    A sample of this format will be:
        content of sample_1<eod>
        content of sample_2<eod>
        ...
        content of sample_n<eod>
        <|pad|><|pad|>...<|pad|>
    """

    def __init__(self, seq_length, model_path):
        self.seq_length = seq_length
        self.model_path = model_path

    def initializer(self):
        # Use Encoder class as a container for global data
        assert self.model_path is not None
        # self.tokenizer = load_tokenizer(model_path, tokenizer_type) #AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        # PackSSTBinEncoder.tokenizer = load_tokenizer(self.model_path, self.tokenizer_type)
        PackSSTBinEncoder.tokenizer = init_tokenizer(self.model_path)

    def _encode_content(self, item, encode_res):
        if "content" in item:
            content = item["content"]
        else:
            content = item["text"]

        item_len = len(content)

        input_ids = self.tokenize_string(content)
        encode_res["input_ids"].append(input_ids)

        return encode_res, item_len

    def _encode_chatml(self, item, encode_res):
        input_ids = []
        item_len = 0
        one_round_content = ""
        for i in range(len(item["chat_rounds"])):
            chat_round = item["chat_rounds"][i]
            role = chat_round["role"]
            content = chat_round["content"]
            content = content if content.endswith("\n") else f"{content}\n"
            if role.lower() == "system":
                continue
            if role.lower() == "human":
                one_round_content = content
            else:
                one_round_content += content
                input_ids += self.tokenize_string(one_round_content)
                item_len += len(one_round_content)

        encode_res["input_ids"].append(input_ids)

        return encode_res, item_len

    def encode(self, item):
        encode_res = {
            "input_ids": [],
        }

        try:
            if item is None:
                encode_res["input_ids"].append([])
                return encode_res, 0

            if "content" in item or "text" in item:
                return self._encode_content(item, encode_res)

            if "chat_rounds" in item:
                return self._encode_chatml(item, encode_res)
        except Exception as e:
            print("####JSON Exception", e, str(item))
            encode_res["input_ids"].append([])
            return encode_res, 0

        raise Exception("Unsupported Format!")

    def tokenize_string(self, text):
        end_marker = [PackSSTBinEncoder.tokenizer.eos_token_id]

        input_ids = []
        try:
            input_ids = PackSSTBinEncoder.tokenizer.encode(text, add_special_tokens=False)
            input_ids = input_ids + end_marker
            return input_ids
        except Exception as e:
            print("####Tokenization Exception:", e, text)
            return []
        except BaseException as e:
            print("####Tokenization BaseException:", e, "Length of text", len(text))
            return []

    def padding(self, data, pad_token_id):
        assert len(data) <= self.seq_length, f"padding sequence: {len(data)} > {self.seq_length}"
        return data + [pad_token_id] * (self.seq_length - len(data))
