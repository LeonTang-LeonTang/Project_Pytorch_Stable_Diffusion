import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

"""
CLIP (Contrastive Language-Image Pre-training)

CLIP connects text and images:
Text: "A cat sitting on a mat" ──┐
                                  ├─→ Compare similarity
Image: [Photo of cat on mat] ────┘

The text encoder that converts text → meaningful vectors

This CLIP code is a general-purpose text encoder that can be used for:

✅ Text-Image matching (original CLIP purpose)
✅ Image captioning
✅ Text-to-image generation (like Stable Diffusion - but CLIP is just one component)
✅ Zero-shot classification
✅ Text generation (potentially)

The Complete Picture:

### Why CLIP + Diffusion is Optimal:
```
┌────────────────────────────────────────────────────┐
│ Input: "A majestic lion in golden sunset"        │
└────────────────────────────────────────────────────┘
                        ↓
┌────────────────────────────────────────────────────┐
│ CLIP Text Encoder (Pre-trained)                   │
│                                                     │
│ What it does:                                      │
│ ✓ Understands "majestic" = regal, impressive      │
│ ✓ Knows "lion" is a big cat                       │
│ ✓ Associates "golden sunset" with warm colors     │
│ ✓ Outputs: [0.2, 0.8, -0.3, 1.5, ...]            │
│                                                     │
│ Cost: FREE (already trained)                       │
└────────────────────────────────────────────────────┘
                        ↓
            768-dimensional vector
                        ↓
┌────────────────────────────────────────────────────┐
│ Diffusion Model (U-Net)                           │
│                                                     │
│ What it does:                                      │
│ ✓ Takes text embedding as guidance                │
│ ✓ Starts with random noise                        │
│ ✓ Gradually denoises, following text guidance     │
│ ✓ Uses cross-attention to check text at each step │
│                                                     │
│ Training: Only this part needs training            │
└────────────────────────────────────────────────────┘
                        ↓
              ┌─────────────────┐
              │  Generated      │
              │  Image of       │
              │  Lion in        │
              │  Sunset         │
              └─────────────────┘
```
### Complete CLIP Text Encoder Architecture Visualization:
```
Input: Token IDs (Batch, Seq_Len)
    ↓
┌──────────────────────────────┐
│ CLIPEmbedding                │
│ ├─ Token Embedding           │
│ └─ Position Embedding        │
└──────────────────────────────┘
    ↓ (Batch, Seq_Len, 768)
┌──────────────────────────────┐
│ CLIPLayer 1                  │
│ ├─ Self-Attention Block      │
│ └─ Feedforward Block         │
└──────────────────────────────┘
    ↓
┌──────────────────────────────┐
│ CLIPLayer 2                  │
└──────────────────────────────┘
    ↓
    ... (layers 3-11)
    ↓
┌──────────────────────────────┐
│ CLIPLayer 12                 │
└──────────────────────────────┘
    ↓ (Batch, Seq_Len, 768)
┌──────────────────────────────┐
│ Final LayerNorm              │
└──────────────────────────────┘
    ↓
Output: Contextualized embeddings
(Batch, Seq_Len, 768)

"""
class CLIPEmbedding(nn.Module):#Converting Words to Vectors
    
    #n_vocab： Vocabulary size (number of unique words/tokens)
    #n_tokens: Maximum sequence length
    #n_embd: Embedding dimension
    def __init__(self, n_vocab:int, n_embd:int, n_tokens: int):
        super().__init__()
        
        #nn.Embedding：It's a lookup table that converts word IDs to vectors
        #Token Embedding (Word Meaning)；Position Embedding (Word Position)
        
        #nn.Parameter:
        # Makes this a learnable parameter
        # Starts at zeros, trained during learning
        # Different from fixed sinusoidal positional encodings (used in original Transformer)
        self.token_embedding =nn.Embedding(n_vocab, n_embd)
        self.position_embedding =nn.Parameter(torch.zeros(n_tokens, n_embd))
        
        """
        e.g.
        # Sentence: "A cat sits"
        token_ids = [49406, 320, 2368, 7718, 49407]  # IDs for [START] A cat sits [END]

        # After token embedding:
        # [49406] → [768 numbers representing START token]
        # [320]   → [768 numbers representing "A"]
        # [2368]  → [768 numbers representing "cat"]
        # [7718]  → [768 numbers representing "sits"]
        # [49407] → [768 numbers representing END token]
        
        # Position 0 (first word):  [0.1, 0.2, -0.3, ...]  ← 768 numbers
        # Position 1 (second word): [0.2, -0.1, 0.4, ...]  ← 768 numbers
        # Position 2 (third word):  [-0.1, 0.3, 0.2, ...]  ← 768 numbers
        # ...
        # Position 76 (last possible position)
        """
        
    def forward(self, tokens):
        # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
        x =self.token_embedding(tokens)
        
        x+=self.position_embedding
        
        return x
    
class CLIPLayer(nn.Module):
    """
     CLIPLayer Complete Flow:
    ```
    Input: (Batch, Seq_Len, 768)
        ↓
    ┌─────────────────────────┐
    │ Self-Attention Block    │
    │ ├─ LayerNorm            │
    │ ├─ Self-Attention       │
    │ └─ Add Residual         │
    └─────────────────────────┘
        ↓ (Batch, Seq_Len, 768)
    ┌─────────────────────────┐
    │ Feedforward Block       │
    │ ├─ LayerNorm            │
    │ ├─ Linear (768→3072)    │
    │ ├─ GELU                 │
    │ ├─ Linear (3072→768)    │
    │ └─ Add Residual         │
    └─────────────────────────┘
        ↓
    Output: (Batch, Seq_Len, 768)
"""
    
    def __init__(self, n_head: int, n_embd: int):
        super().__init__()
        
        #LayerNorm: Normalizes features to have mean=0, std=1:
        #LayerNorm(x) = (x - mean(x)) / sqrt(var(x) + ε) * γ + β
        """
        γ (gamma): Learnable scale
        β (beta): Learnable shift
        ε (epsilon): Small constant for numerical stability

        Why needed?

        Stabilizes training (prevents exploding/vanishing values)
        """
        self.layernorm_1 = nn.LayerNorm(n_embd)
        self.attention =SelfAttention(n_head, n_embd)
        self.layernorm_2 = nn.LayerNorm(n_embd)
        #.Feedforward Network
        """
        **Two linear layers with expansion**:
        ```
        768 → 3072 → 768
            ↑
        4× larger!
        ```

        **Why expand then compress?**
        - Creates a "bottleneck" architecture
        - Allows non-linear transformations
        - Increases model capacity
        """
        self.linear_1 = nn.Linear(n_embd, 4*n_embd)
        self.linear_2 = nn.Linear(4* n_embd, n_embd)
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # (Batch_Size, Seq_Len, Dim)
        
        residue =x
        ## SELF ATTENTION
        
        x = self.layernorm_1(x)
        """
        Why causal mask in CLIP text encoder?
            Follows GPT-style autoregressive pattern
            Each word only attends to previous words and itself
            Useful for text generation tasks
            Prevents "looking ahead" (cheating!)
        """
        x=self.attention(x, causal_mask=True) 
        #With causal mask, the model can be used for: A. Text encoding (CLIP's main use) B. Text generation (potential future use)
        
        x+=residue
        
        #FEEDFORWARD LAYER
        
        residue =x
        x =self.layernorm_2(x)
        
        x=self.linear_1(x)
        """
        What is GELU? (Gaussian Error Linear Unit)  
            GELU(x) ≈ x * sigmoid(1.702 * x)
            
        Why GELU?
            - Smooth, differentiable activation
            - Better than ReLU for Transformers
            - Allows small negative values (unlike ReLU)
        e.g. 
        x = 2.0
        gelu = 2.0 * sigmoid(1.702 * 2.0)
            = 2.0 * sigmoid(3.404)
            = 2.0 * 0.968
            = 1.936
        
        """
        # GELU activation funciton
        x=x*torch.sigmoid(1.702*x) 
        x =self.linear_2(x)
        
        x+=residue
        
        return x
    
class CLIP(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.embedding =CLIPEmbedding(49408, 768,77)
        
        #Why 12 layers? More layers = deeper understanding; 12 is a common choice (BERT-base also uses 12)
        #12 attention heads; 768-dimensional embeddings
        self.layers =nn.ModuleList([
            CLIPLayer(12, 768) for i in range(12)
        ])
        
        self.layernorm = nn.LayerNorm(768)
        
    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        # Example input:tokens = torch.tensor([[49406, 320, 2368, 7718, 49407]])
        # Shape: (1, 5) - "A cat sits"
        tokens =tokens.type(torch.long)
        
        # Step 1: Embedding # Shape: (1, 5, 768)
        #(Batch_Size, Seq_Len) -> (Batch_Szie, Seq_Len, Dim)
        state =self.embedding(tokens)
        
        # Step 2: Process through 12 layers
        for layer in self.layers: # Loops 12 times
            state =layer(state)
            # Each layer: attention + feedforward + residuals
            # Shape stays: (1, 5, 768)
        
        # Step 3: Final normalization # Shape: (1, 5, 768) # Normalized representations
        #(Batch_Size, Seq_Len, Dim)
        output =self.layernorm(state)
        
        return output
"""
You're asking:  Why Do We Need CLIP in Diffusion Models?

Current (Stable Diffusion):
Text → [CLIP Encoder] → Text Embedding → [Diffusion Model] → Image

Why not simpler?
Text → [Diffusion Model] → Image directly?
```

**Short answer**: The diffusion model **CAN'T understand text directly**. It only understands numbers/vectors. CLIP translates text into a form the diffusion model can use.

---

## Part 1: What Diffusion Models Actually Do

### Diffusion Models are Image Experts, Not Text Experts

**A diffusion model** (like the U-Net in Stable Diffusion) is trained to:
1. Take a noisy image
2. Predict and remove noise
3. Gradually reveal a clean image
```
Pure Diffusion (without text):
Random noise → [Denoise] → [Denoise] → [Denoise] → Random image

Problem: No control! Generates random images.
```

**Visual**:
```
Step 0: ▓▓▓▓▓▓▓▓ (pure noise)
Step 1: ▓▓▓▒▒▓▓▓ (slightly less noise)
Step 2: ▓▒▒░░▒▓▓ (more structure)
Step 3: ▒░░  ░▒▒ (image emerging)
Step 4: ░     ░░ (clean image)

But what image? Random! No way to specify "make a cat"
```

---

### The Problem: How to Control Generation?

**We want**:
```
Input: "A cat sitting on a mat"
Output: Image of exactly that!
```

**But diffusion model speaks "image language"** (pixels, features, noise), not "text language" (words)!

**Analogy**:
```
You (English speaker): "I want a cat image"
Diffusion Model (only speaks Japanese): "???"

Need translator:
You → [Translator] → Japanese instructions → Diffusion Model → Cat image
     ↑
   This is CLIP's role!
```

---

## Part 2: Why Can't Diffusion Model Learn Text Directly?

### Reason 1: Different Data Modalities

**Text and images are fundamentally different**:
```
Text:
- Discrete (words, tokens)
- Sequential (order matters)
- Symbolic (abstract meaning)
- Small data size (bytes)

Images:
- Continuous (pixel values)
- Spatial (2D grid)
- Perceptual (visual features)
- Large data size (megabytes)
"""
        
