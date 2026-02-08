"""
Flask web application for English-to-Thai machine translation.
Loads the trained Transformer model and provides a simple UI
where users can type an English sentence and get the Thai translation.
"""
import os
import sys
import json
import re
from functools import partial

import torch
import torch.nn as nn

from flask import Flask, render_template, request, jsonify

# add parent dir so we can import pythainlp
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from pythainlp.tokenize import word_tokenize as th_tokenize

# paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, '..', 'model')

app = Flask(__name__)

# ---- Tokenizer helpers (same as in notebook) ----

def _spacy_tokenize(x, spacy):
    return [tok.text for tok in spacy.tokenizer(x)]

_patterns = [
    r"\'", r"\"", r"\.", r"<br \/>",
    r",", r"\(", r"\)", r"\!",
    r"\?", r"\;", r"\:", r"\s+",
]
_replacements = [
    " '  ", "", " . ", " ",
    " , ", " ( ", " ) ", " ! ",
    " ? ", " ", " ", " ",
]
_patterns_dict = list(
    (re.compile(p), r) for p, r in zip(_patterns, _replacements)
)

def _basic_english_normalize(line):
    line = line.lower()
    for pattern_re, replaced_str in _patterns_dict:
        line = pattern_re.sub(replaced_str, line)
    return line.split()


# ---- Vocab class (same as notebook) ----

class Vocab:
    def __init__(self, stoi, itos, default_index=None):
        self._itos = itos
        self._stoi = stoi
        self._default_index = default_index

    def __len__(self):
        return len(self._itos)

    def __getitem__(self, token):
        if self._default_index is not None:
            return self._stoi.get(token, self._default_index)
        return self._stoi[token]

    def set_default_index(self, index):
        self._default_index = index

    def lookup_indices(self, tokens):
        return [self[t] for t in tokens]

    def lookup_tokens(self, indices):
        return [self._itos[i] for i in indices]

    def get_itos(self):
        return list(self._itos)

    def get_stoi(self):
        return dict(self._stoi)


# ---- Model classes (same architecture as notebook) ----

class GeneralAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        assert hid_dim % n_heads == 0
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        attention = torch.softmax(energy, dim=-1)
        x = torch.matmul(self.dropout(attention), V)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.hid_dim)
        x = self.fc_o(x)
        return x, attention


class AdditiveAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        assert hid_dim % n_heads == 0
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        self.W1 = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.W2 = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.v = nn.Linear(self.head_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        Q_expanded = Q.unsqueeze(3)
        K_expanded = K.unsqueeze(2)
        energy = self.v(torch.tanh(self.W1(K_expanded) + self.W2(Q_expanded))).squeeze(-1)
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        attention = torch.softmax(energy, dim=-1)
        x = torch.matmul(self.dropout(attention), V)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.hid_dim)
        x = self.fc_o(x)
        return x, attention


class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(hid_dim, pf_dim)
        self.fc2 = nn.Linear(pf_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device, attention_class):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = attention_class(hid_dim, n_heads, dropout, device)
        self.feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        _src, _ = self.self_attention(src, src, src, src_mask)
        src = self.self_attn_layer_norm(src + self.dropout(_src))
        _src = self.feedforward(src)
        src = self.ff_layer_norm(src + self.dropout(_src))
        return src


class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, device, attention_class, max_length=200):
        super().__init__()
        self.device = device
        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        self.layers = nn.ModuleList([
            EncoderLayer(hid_dim, n_heads, pf_dim, dropout, device, attention_class)
            for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, src, src_mask):
        batch_size = src.shape[0]
        src_len = src.shape[1]
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))
        for layer in self.layers:
            src = layer(src, src_mask)
        return src


class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device, attention_class):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = attention_class(hid_dim, n_heads, dropout, device)
        self.encoder_attention = attention_class(hid_dim, n_heads, dropout, device)
        self.feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))
        _trg = self.feedforward(trg)
        trg = self.ff_layer_norm(trg + self.dropout(_trg))
        return trg, attention


class Decoder(nn.Module):
    def __init__(self, output_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, device, attention_class, max_length=200):
        super().__init__()
        self.device = device
        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        self.layers = nn.ModuleList([
            DecoderLayer(hid_dim, n_heads, pf_dim, dropout, device, attention_class)
            for _ in range(n_layers)
        ])
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))
        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)
        output = self.fc_out(trg)
        return output, attention


class Seq2SeqTransformer(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, trg_pad_idx, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)
        return output, attention


# ---- Load model and vocab ----

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load config
with open(os.path.join(MODEL_DIR, 'config.json'), 'r') as f:
    config = json.load(f)

# load vocab
with open(os.path.join(MODEL_DIR, 'vocab.json'), 'r', encoding='utf-8') as f:
    vocab_data = json.load(f)

src_vocab = Vocab(vocab_data['src_stoi'], vocab_data['src_itos'], default_index=config['unk_idx'])
trg_vocab = Vocab(vocab_data['trg_stoi'], vocab_data['trg_itos'], default_index=config['unk_idx'])

# set up tokenizers
try:
    import spacy
    spacy_en = spacy.load('en_core_web_sm')
    en_tokenizer = lambda x: [tok.text for tok in spacy_en.tokenizer(x)]
except Exception:
    en_tokenizer = _basic_english_normalize

th_tokenizer = lambda x: th_tokenize(x, engine='newmm')

# pick attention class
attention_map = {
    'General Attention': GeneralAttentionLayer,
    'Additive Attention': AdditiveAttentionLayer,
}
attn_class = attention_map[config['attention_type']]

# build and load model
enc = Encoder(config['input_dim'], config['hid_dim'], config['n_layers'], config['n_heads'],
              config['pf_dim'], config['dropout'], device, attn_class, max_length=512)
dec = Decoder(config['output_dim'], config['hid_dim'], config['n_layers'], config['n_heads'],
              config['pf_dim'], config['dropout'], device, attn_class, max_length=512)
model = Seq2SeqTransformer(enc, dec, config['pad_idx'], config['pad_idx'], device).to(device)
model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'best_model.pt'), map_location=device))
model.eval()

print(f"Model loaded: {config['attention_type']} on {device}")


def translate(sentence, max_len=50):
    """Translate an English sentence to Thai."""
    # tokenize and numericalize
    tokens = en_tokenizer(sentence)
    indices = [config['sos_idx']] + src_vocab.lookup_indices(tokens) + [config['eos_idx']]
    src_tensor = torch.LongTensor(indices).unsqueeze(0).to(device)

    src_mask = model.make_src_mask(src_tensor)
    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    trg_indices = [config['sos_idx']]
    for _ in range(max_len):
        trg_tensor = torch.LongTensor(trg_indices).unsqueeze(0).to(device)
        trg_mask = model.make_trg_mask(trg_tensor)
        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
        pred_token = output.argmax(2)[:, -1].item()
        trg_indices.append(pred_token)
        if pred_token == config['eos_idx']:
            break

    trg_tokens = trg_vocab.lookup_tokens(trg_indices)
    # remove special tokens for display
    translation = ''.join([t for t in trg_tokens if t not in ['<sos>', '<eos>', '<pad>', '<unk>']])
    return translation, trg_tokens, tokens


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/translate', methods=['POST'])
def translate_api():
    data = request.get_json()
    sentence = data.get('sentence', '')
    if not sentence.strip():
        return jsonify({'error': 'Please enter a sentence.'}), 400

    translation, trg_tokens, src_tokens = translate(sentence)
    return jsonify({
        'source': sentence,
        'source_tokens': src_tokens,
        'translation': translation,
        'translation_tokens': trg_tokens,
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
