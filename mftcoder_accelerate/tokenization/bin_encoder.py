from transformers import AutoTokenizer
from tokenizer import init_tokenizer


class SSTBinEncoder:
    """
    A sample of this format will be:
        content of sample_1<eod>
        content of sample_2<eod>
        ...
        content of sample_n<eod>
        <|pad|><|pad|>...<|pad|>
    """
    tokenizer = None

    def __init__(self, seq_length, model_path):
        self.seq_length = seq_length
        self.model_path = model_path

    def initializer(self):
        # Use Encoder class as a container for global data
        assert self.model_path is not None
        SSTBinEncoder.tokenizer = init_tokenizer(self.model_path)

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
        end_marker = [SSTBinEncoder.tokenizer.eos_token_id]

        input_ids = []
        try:
            input_ids = SSTBinEncoder.tokenizer.encode(text, add_special_tokens=False)
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
