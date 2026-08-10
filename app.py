import os
import json
import numpy as np

import gradio as gr
import spaces
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# ENVIRONMENT
# ============================================================

print("=" * 60)
print("MCQ Solver - Startup")
print("=" * 60)

print("Gradio version:", gr.__version__)
print("PyTorch version:", torch.__version__)
print("CUDA available at startup:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("=" * 60)


# ============================================================
# DEVICE
# ============================================================

CPU_DEVICE = torch.device("cpu")


# ============================================================
# MODEL CLASSES
# ============================================================

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super().__init__()

        pe = torch.zeros(
            max_len,
            d_model
        )

        position = torch.arange(
            0,
            max_len,
            dtype=torch.float
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                2
            ).float()
            * (-np.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(
            position * div_term
        )

        pe[:, 1::2] = torch.cos(
            position * div_term
        )

        pe = pe.unsqueeze(0)

        self.register_buffer(
            "pe",
            pe
        )

    def forward(self, x):

        return x + self.pe[:, :x.size(1), :]


class TransformerBlock(nn.Module):

    def __init__(
        self,
        d_model,
        num_heads,
        d_ff,
        dropout=0.1
    ):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout
        )

        self.norm1 = nn.LayerNorm(
            d_model
        )

        self.norm2 = nn.LayerNorm(
            d_model
        )

        self.ff = nn.Sequential(

            nn.Linear(
                d_model,
                d_ff
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                d_ff,
                d_model
            ),

            nn.Dropout(
                dropout
            )
        )

    def forward(self, x):

        attn_output, _ = self.attention(
            x,
            x,
            x
        )

        x = self.norm1(
            x + attn_output
        )

        ff_output = self.ff(x)

        x = self.norm2(
            x + ff_output
        )

        return x


class ScratchModel(nn.Module):

    def __init__(self, config):

        super().__init__()

        self.embedding = nn.Embedding(
            config["vocab_size"],
            config["embedding_dim"],
            padding_idx=0
        )

        self.positional = PositionalEncoding(
            config["embedding_dim"],
            config["max_length"]
        )

        self.dropout = nn.Dropout(
            config["dropout"]
        )

        self.layers = nn.ModuleList([

            TransformerBlock(
                config["embedding_dim"],
                8,
                config["hidden_dim"],
                config["dropout"]
            )

            for _ in range(
                config["num_layers"]
            )
        ])

        self.classifier = nn.Sequential(

            nn.Linear(
                config["embedding_dim"] * 3,
                config["hidden_dim"]
            ),

            nn.BatchNorm1d(
                config["hidden_dim"]
            ),

            nn.GELU(),

            nn.Dropout(
                config["dropout"]
            ),

            nn.Linear(
                config["hidden_dim"],
                config["hidden_dim"] // 2
            ),

            nn.GELU(),

            nn.Dropout(
                config["dropout"]
            ),

            nn.Linear(
                config["hidden_dim"] // 2,
                5
            )
        )

    def forward(self, input_ids):

        x = self.embedding(
            input_ids
        )

        x = self.positional(x)

        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x)

        # ----------------------------------------------------
        # IMPORTANT:
        # Keep these pooling operations exactly compatible
        # with the model used during training.
        # ----------------------------------------------------

        mean_pooled = x.mean(
            dim=1
        )

        max_pooled, _ = x.max(
            dim=1
        )

        weights = F.softmax(
            x.mean(
                dim=1,
                keepdim=True
            ),
            dim=1
        )

        weighted_pool = torch.sum(
            x * weights,
            dim=1
        )

        pooled = torch.cat(
            [
                mean_pooled,
                max_pooled,
                weighted_pool
            ],
            dim=1
        )

        return self.classifier(
            pooled
        )


# ============================================================
# TOKENIZER
# ============================================================

class SimpleTokenizer:

    def __init__(
        self,
        word2idx,
        max_length
    ):

        self.word2idx = word2idx
        self.max_length = max_length

    def encode(self, text):

        words = str(
            text
        ).lower().split()

        # Special tokens
        # PAD = 0
        # UNK = 1
        # SOS = 2
        # EOS = 3

        sequence = [2]

        for word in words[
            :self.max_length - 2
        ]:

            sequence.append(
                self.word2idx.get(
                    word,
                    1
                )
            )

        sequence.append(3)

        # Padding
        if len(sequence) < self.max_length:

            sequence += [
                0
            ] * (
                self.max_length
                - len(sequence)
            )

        else:

            sequence = sequence[
                :self.max_length
            ]

        return sequence


# ============================================================
# MODEL PATH
# ============================================================

MODEL_PATH = os.path.join(
    "models",
    "scratch_huggingface"
)

CONFIG_FILE = os.path.join(
    MODEL_PATH,
    "config.json"
)

TOKENIZER_FILE = os.path.join(
    MODEL_PATH,
    "tokenizer_config.json"
)

WEIGHTS_FILE = os.path.join(
    MODEL_PATH,
    "pytorch_model.bin"
)


# ============================================================
# GLOBAL MODEL VARIABLES
# ============================================================

model = None
tokenizer = None
config = None


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    global model
    global tokenizer
    global config

    print("🔄 Loading model...")

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    required_files = [
        CONFIG_FILE,
        TOKENIZER_FILE,
        WEIGHTS_FILE
    ]

    for file_path in required_files:

        if not os.path.exists(file_path):

            raise FileNotFoundError(
                f"Required model file not found: "
                f"{file_path}"
            )

    print(
        f"✅ Found model at: {MODEL_PATH}"
    )

    # --------------------------------------------------------
    # Load config
    # --------------------------------------------------------

    with open(
        CONFIG_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        config = json.load(f)

    print(
        f"   Layers: "
        f"{config.get('num_layers')}"
    )

    print(
        f"   Embedding Dim: "
        f"{config.get('embedding_dim')}"
    )

    print(
        f"   Hidden Dim: "
        f"{config.get('hidden_dim')}"
    )

    print(
        f"   Vocabulary: "
        f"{config.get('vocab_size')}"
    )

    print(
        f"   Max Length: "
        f"{config.get('max_length')}"
    )

    # --------------------------------------------------------
    # Load tokenizer
    # --------------------------------------------------------

    with open(
        TOKENIZER_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        tokenizer_config = json.load(f)

    tokenizer = SimpleTokenizer(
        tokenizer_config["word2idx"],
        tokenizer_config["max_length"]
    )

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = ScratchModel(
        config
    )

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    print("🔄 Loading model weights...")

    checkpoint = torch.load(
        WEIGHTS_FILE,
        map_location="cpu"
    )

    if isinstance(
        checkpoint,
        dict
    ) and "model_state_dict" in checkpoint:

        state_dict = checkpoint[
            "model_state_dict"
        ]

    else:

        state_dict = checkpoint

    # --------------------------------------------------------
    # Load weights
    # --------------------------------------------------------

    model.load_state_dict(
        state_dict
    )

    model.eval()

    # Always keep model on CPU initially.
    # ZeroGPU provides GPU only when @spaces.GPU
    # function is executing.

    model.to(
        CPU_DEVICE
    )

    print("✅ Model loaded successfully!")

    return model


# ============================================================
# LOAD MODEL AT STARTUP
# ============================================================

try:

    load_model()

except Exception as e:

    print(
        "❌ MODEL LOADING FAILED"
    )

    print(
        f"Error: {e}"
    )

    # Do NOT create a random fallback model.
    raise


# ============================================================
# PREDICTION
# ============================================================

@spaces.GPU
def predict(
    prompt,
    option_a,
    option_b,
    option_c,
    option_d,
    option_e
):

    global model

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if not prompt or not prompt.strip():

        return (
            "⚠️ **Please enter a question.**"
        )

    options = [
        option_a,
        option_b,
        option_c,
        option_d,
        option_e
    ]

    if not all(
        option and option.strip()
        for option in options
    ):

        return (
            "⚠️ **Please fill all 5 options.**"
        )

    try:

        # ----------------------------------------------------
        # GPU device
        # ----------------------------------------------------

        device = torch.device(
            "cuda"
        )

        print(
            "🚀 GPU inference started"
        )

        # ----------------------------------------------------
        # Move model to GPU
        # ----------------------------------------------------

        model = model.to(
            device
        )

        model.eval()

        # ----------------------------------------------------
        # Build input
        # ----------------------------------------------------

        text = (
            f"{prompt} [SEP] "
            + " [SEP] ".join(options)
        )

        # ----------------------------------------------------
        # Tokenize
        # ----------------------------------------------------

        input_ids = tokenizer.encode(
            text
        )

        input_tensor = torch.tensor(
            [input_ids],
            dtype=torch.long,
            device=device
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        with torch.inference_mode():

            outputs = model(
                input_tensor
            )

            probabilities = F.softmax(
                outputs,
                dim=1
            )

            top3 = torch.topk(
                probabilities,
                k=3,
                dim=1
            ).indices[0]

        # ----------------------------------------------------
        # Move probabilities to CPU
        # ----------------------------------------------------

        probabilities_cpu = (
            probabilities[0]
            .detach()
            .cpu()
        )

        # ----------------------------------------------------
        # Results
        # ----------------------------------------------------

        letters = [
            "A",
            "B",
            "C",
            "D",
            "E"
        ]

        output = (
            "## 🏆 Top 3 Predictions\n\n"
        )

        for rank, idx in enumerate(
            top3.detach().cpu().tolist(),
            start=1
        ):

            confidence = (
                float(
                    probabilities_cpu[idx]
                ) * 100
            )

            badge = [
                "🥇",
                "🥈",
                "🥉"
            ][rank - 1]

            output += f"""
### {badge} Rank {rank}

**Option:** `{letters[idx]}`

**Answer:** {options[idx]}

**Confidence:** `{confidence:.2f}%`

---
"""

        print(
            "✅ GPU inference completed"
        )

        return output

    except Exception as e:

        print(
            f"❌ Prediction error: {e}"
        )

        return (
            f"⚠️ **Prediction error:**\n\n"
            f"`{str(e)}`"
        )

    finally:

        # ----------------------------------------------------
        # Move model back to CPU after ZeroGPU execution.
        # ----------------------------------------------------

        try:

            model = model.to(
                CPU_DEVICE
            )

            if torch.cuda.is_available():

                torch.cuda.empty_cache()

        except Exception as cleanup_error:

            print(
                "⚠️ GPU cleanup warning:",
                cleanup_error
            )


# ============================================================
# CLEAR
# ============================================================

def clear_all():

    return (
        "",
        "",
        "",
        "",
        "",
        "",
        ""
    )


# ============================================================
# GRADIO UI
# ============================================================

def create_interface():

    with gr.Blocks(
        title="MCQ Solver"
    ) as demo:

        # ----------------------------------------------------
        # Header
        # ----------------------------------------------------

        gr.Markdown(
            """
# 🧠 MCQ Solver

### Scratch Transformer Model

Enter a multiple-choice question and five options.
The model will return the **Top 3 predictions**.
"""
        )

        # ----------------------------------------------------
        # Question
        # ----------------------------------------------------

        prompt = gr.Textbox(
            label="Question",
            placeholder=(
                "Enter your MCQ question here..."
            ),
            lines=4
        )

        # ----------------------------------------------------
        # Options
        # ----------------------------------------------------

        gr.Markdown(
            "### 📝 Answer Options"
        )

        with gr.Row():

            with gr.Column():

                option_a = gr.Textbox(
                    label="A",
                    placeholder="Option A"
                )

                option_b = gr.Textbox(
                    label="B",
                    placeholder="Option B"
                )

                option_c = gr.Textbox(
                    label="C",
                    placeholder="Option C"
                )

            with gr.Column():

                option_d = gr.Textbox(
                    label="D",
                    placeholder="Option D"
                )

                option_e = gr.Textbox(
                    label="E",
                    placeholder="Option E"
                )

        # ----------------------------------------------------
        # Buttons
        # ----------------------------------------------------

        with gr.Row():

            solve_button = gr.Button(
                "🧠 Solve",
                variant="primary"
            )

            clear_button = gr.Button(
                "🗑️ Clear"
            )

        # ----------------------------------------------------
        # Output
        # ----------------------------------------------------

        gr.Markdown(
            "### 📊 Prediction"
        )

        result = gr.Markdown()

        # ----------------------------------------------------
        # Model Information
        # ----------------------------------------------------

        with gr.Accordion(
            "⚙️ Model Information",
            open=False
        ):

            model_info = gr.Markdown(
                f"""
**Architecture:** Scratch Transformer

**Layers:** `{config.get("num_layers", "N/A")}`

**Embedding Dimension:** `{config.get("embedding_dim", "N/A")}`

**Hidden Dimension:** `{config.get("hidden_dim", "N/A")}`

**Vocabulary Size:** `{config.get("vocab_size", "N/A")}`

**Maximum Sequence Length:** `{config.get("max_length", "N/A")}`

**Number of Classes:** `5`

**Startup CUDA:** `{torch.cuda.is_available()}`

**GPU:** ZeroGPU via `@spaces.GPU`
"""
            )

        # ----------------------------------------------------
        # Instructions
        # ----------------------------------------------------

        with gr.Accordion(
            "📖 How to Use",
            open=False
        ):

            gr.Markdown(
                """
1. Enter the MCQ question.
2. Enter all five options.
3. Click **🧠 Solve**.
4. Wait for GPU allocation.
5. View the Top 3 predictions.
"""
            )

        # ----------------------------------------------------
        # Events
        # ----------------------------------------------------

        solve_button.click(
            fn=predict,
            inputs=[
                prompt,
                option_a,
                option_b,
                option_c,
                option_d,
                option_e
            ],
            outputs=result
        )

        clear_button.click(
            fn=clear_all,
            inputs=[],
            outputs=[
                prompt,
                option_a,
                option_b,
                option_c,
                option_d,
                option_e,
                result
            ]
        )

    return demo


# ============================================================
# CREATE APP
# ============================================================

demo = create_interface()


# ============================================================
# LAUNCH
# ============================================================

if __name__ == "__main__":

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        ssr_mode=False
    )