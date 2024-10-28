
import threading
import fcntl
import json

class JSONLWriter():
    """
    A writer used to save jsonl lines into a file.
    """
    def __init__(self, output_path, dataset_name):
        self.output_path = output_path
        self.out_file = open(output_path, 'w')
        self.cache = []
        self.cache_size = 4096
        self.dataset_name = dataset_name
        self.index = 0

    def pack_into_jsonl(self, line_text):
        new_item = {
            "data_name": self.dataset_name,
            "id": self.index,
            "content": line_text
        }

        return new_item


    def add_item(self, line_text):
        if len(self.cache) >= self.cache_size:
            self.flush()
        
        item = self.pack_into_jsonl(line_text)
        self.cache.append(json.dumps(item))
        self.index += 1

    
    def flush(self):
        content = '\n'.join(self.cache)
        fcntl.flock(self.out_file, fcntl.LOCK_EX)
        self.out_file.write(f'{content}\n')
        fcntl.flock(self.out_file, fcntl.LOCK_UN)
        self.cache = [] 
