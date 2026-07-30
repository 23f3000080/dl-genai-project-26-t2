import streamlit as st
import torch
import json
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# MODEL CLASSES (Copy from training)
# ============================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_output)
        ff_output = self.ff(x)
        x = self.norm2(x + ff_output)
        return x

class ScratchModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config['vocab_size'], config['embedding_dim'], padding_idx=0)
        self.positional = PositionalEncoding(config['embedding_dim'], config['max_length'])
        self.dropout = nn.Dropout(config['dropout'])
        self.layers = nn.ModuleList([
            TransformerBlock(config['embedding_dim'], 8, config['hidden_dim'], config['dropout'])
            for _ in range(config['num_layers'])
        ])
        self.classifier = nn.Sequential(
            nn.Linear(config['embedding_dim'] * 3, config['hidden_dim']),
            nn.BatchNorm1d(config['hidden_dim']),
            nn.GELU(),
            nn.Dropout(config['dropout']),
            nn.Linear(config['hidden_dim'], config['hidden_dim'] // 2),
            nn.GELU(),
            nn.Dropout(config['dropout']),
            nn.Linear(config['hidden_dim'] // 2, 5)
        )
    
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.positional(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x)
        mean_pooled = x.mean(dim=1)
        max_pooled, _ = x.max(dim=1)
        weighted_pool = torch.sum(x * F.softmax(x.mean(dim=1, keepdim=True), dim=1), dim=1)
        pooled = torch.cat([mean_pooled, max_pooled, weighted_pool], dim=1)
        return self.classifier(pooled)

# ============================================================
# TOKENIZER
# ============================================================

class SimpleTokenizer:
    def __init__(self, word2idx, max_length):
        self.word2idx = word2idx
        self.max_length = max_length
        
    def encode(self, text):
        words = str(text).lower().split()
        seq = [2]  # SOS token
        for w in words[:self.max_length-2]:
            seq.append(self.word2idx.get(w, 1))  # 1 = UNK
        seq.append(3)  # EOS token
        seq = seq + [0] * (self.max_length - len(seq))  # 0 = PAD
        return seq

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    model_path = "models/scratch_huggingface"
    
    # Load config
    with open(f"{model_path}/config.json", 'r') as f:
        config = json.load(f)
    
    # Load tokenizer
    with open(f"{model_path}/tokenizer_config.json", 'r') as f:
        tokenizer_config = json.load(f)
    
    # Create model
    model = ScratchModel(config)
    
    # Load weights
    checkpoint = torch.load(f"{model_path}/pytorch_model.bin", map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create tokenizer
    tokenizer = SimpleTokenizer(
        tokenizer_config['word2idx'],
        tokenizer_config['max_length']
    )
    
    return model, tokenizer, config

# ============================================================
# PREDICT
# ============================================================

def predict(prompt, options, model, tokenizer):
    # Combine text
    text = f"{prompt} [SEP] " + " [SEP] ".join(options)
    input_ids = tokenizer.encode(text)
    
    # Predict
    with torch.no_grad():
        outputs = model(torch.tensor([input_ids]))
        probs = F.softmax(outputs, dim=1)
        top3 = torch.argsort(probs, descending=True)[0][:3]
    
    # Results
    letters = ['A', 'B', 'C', 'D', 'E']
    results = []
    for idx in top3:
        results.append({
            'option': letters[idx],
            'text': options[idx],
            'confidence': float(probs[0][idx]) * 100
        })
    
    return results

# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(page_title="MCQ Solver", page_icon="🧠")

st.title("🧠 MCQ Solver")
st.markdown("### Built with Scratch Transformer Model")

# Load model
with st.spinner("Loading model..."):
    model, tokenizer, config = load_model()

st.success("✅ Model Ready!")

# Input
prompt = st.text_area("Question:", height=80)

col1, col2 = st.columns(2)
with col1:
    a = st.text_input("A")
    b = st.text_input("B")
    c = st.text_input("C")
with col2:
    d = st.text_input("D")
    e = st.text_input("E")

options = [a, b, c, d, e]

# Predict
if st.button("Solve", type="primary"):
    if prompt and all(options):
        results = predict(prompt, options, model, tokenizer)
        
        st.markdown("---")
        st.markdown("### Top 3 Predictions")
        
        cols = st.columns(3)
        colors = ['#4CAF50', '#FF9800', '#f44336']
        
        for i, (col, res) in enumerate(zip(cols, results)):
            with col:
                st.markdown(f"""
                <div style="background:{colors[i]}; padding:15px; border-radius:10px; 
                            text-align:center; color:white;">
                    <h1>{res['option']}</h1>
                    <p>{res['text'][:40]}</p>
                    <h2>{res['confidence']:.1f}%</h2>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.warning("Please fill all fields")

# Sidebar
with st.sidebar:
    st.markdown("### Model Info")
    st.write(f"Layers: {config['num_layers']}")
    st.write(f"Embedding: {config['embedding_dim']}")
    st.write(f"Vocab: {config['vocab_size']}")
    
    st.markdown("---")
    st.markdown("### How to use")
    st.write("1. Enter question")
    st.write("2. Fill 5 options")
    st.write("3. Click Solve")

st.markdown("---")
st.caption("Made with PyTorch & Streamlit")