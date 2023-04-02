# pytorch_bert


## 动机
tranformers库中的bert代码为了兼容各种例外情况写的太复杂了，不利于模型魔改，所以简化了一下其代码

## 用法

里面实现了BertModel、BertForMaskedLM这两个比较常用了类，这两个类可以加载transformers中BertModel、BertForMaskedLM保存的参数。
例如：
```python
from transformers import BertForMaskedLM
from modeling import BertForMaskedLM as Bert4MLM
'''
save
'''
bert=BertForMaskedLM.from_pretrained('chinese-roberta-wwm-ext')
torch.save(bert.state_dict(),'bert.bin')

'''
load
'''

cfg={
  "attention_probs_dropout_prob": 0.1,
  "bos_token_id": 0,
  "directionality": "bidi",
  "eos_token_id": 2,
  "gradient_checkpointing": False,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "output_past": True,
  "pad_token_id": 0,
  "pooler_fc_size": 768,
  "pooler_num_attention_heads": 12,
  "pooler_num_fc_layers": 3,
  "pooler_size_per_head": 128,
  "pooler_type": "first_token_transform",
  "position_embedding_type": "absolute",
  "type_vocab_size": 2,
  "vocab_size": 21128
  }
class Config(dict):
    def __getattribute__(self,attr):
        return self[attr]
cfg=Config(cfg)

model=Bert4MLM(cfg)
model.load_state_dict(torch.load('bert.bin'))

```

