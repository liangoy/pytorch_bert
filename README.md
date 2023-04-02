# pytorch_bert


## 动机
tranformers库中的bert代码为了兼容各种例外情况写的太复杂了，不利于模型魔改，所以简化了一下其代码

## 用法

里面实现了BertModel、BertForMaskedLM这两个比较常用了类，这两个类可以加载transformers中BertModel、BertForMaskedLM保存的参数。
例如：
```python
from transformers import BertForMaskedLM
from modeling import BertForMaskedLM as B4MLM
'''
save
'''
bert=BertForMaskedLM.from_pretrained('chinese-roberta-wwm-ext')
torch.save(bert.state_dict(),'bert.bin')

'''
load
'''
model=B4MLM()
model.load_state_dict(torch.load('bert.bin'))

```

