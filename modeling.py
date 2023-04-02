import torch
import math
   
def gelu(x):
    "Implementation of the gelu activation function by Hugging Face"
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
    

class BertEmbeddings(torch.nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = torch.nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = torch.nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = torch.nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = torch.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

    def forward(self,input_ids) :
        
        input_shape = input_ids.size()
        seq_length = input_shape[1]
        inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings.weight[0].unsqueeze(0).unsqueeze(0)
        position_embeddings = self.position_embeddings.weight[:seq_length].unsqueeze(0)

        embeddings = inputs_embeds +token_type_embeddings+position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(torch.nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = torch.nn.Linear(config.hidden_size, self.all_head_size)
        self.key = torch.nn.Linear(config.hidden_size, self.all_head_size)
        self.value = torch.nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = torch.nn.Dropout(config.attention_probs_dropout_prob)
        
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self,hidden_states,attention_mask=None):
        
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        attention_probs = torch.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
            
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        return context_layer


class BertAttention(torch.nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self,hidden_states,attention_mask=None):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
        )
        attention_output = self.output(self_outputs, hidden_states)
        return attention_output

class BertSelfOutput(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = torch.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertIntermediate(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = torch.nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = gelu

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states



class BertOutput(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = torch.nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = torch.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self,hidden_states,attention_mask= None):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
        )
        attention_output = self_attention_outputs
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertEncoder(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = torch.nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(self,hidden_states,attention_mask=None):
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states,attention_mask)
        return hidden_states

class BertModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

    def forward(self,input_ids,attention_mask = None):
        embedding_output = self.embeddings(
            input_ids=input_ids,
        )
        sequence_output = self.encoder(
            embedding_output,
            attention_mask=attention_mask
        )
        return sequence_output

class BertPredictionHeadTransform(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = gelu
        self.LayerNorm = torch.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class BertLMPredictionHead(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = torch.nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias
    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states

class BertOnlyMLMHead(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output: torch.Tensor):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores

class BertForMaskedLM(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config)
        self.cls.predictions.decoder.weight=self.bert.embeddings.word_embeddings.weight
    def forward(self,input_ids,attention_mask=None):
        sequence_output = self.bert(
            input_ids,
            attention_mask=attention_mask,
        )
        prediction_scores = self.cls(sequence_output)
        return prediction_scores
