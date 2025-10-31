import torch 
from torch import nn
from torch.nn import functional as F
import math

"""
    What Problem Does Self-Attention Solve?
    
    Goal: Understand relationships between different parts of a sequence.
    Key idea: Look at ALL parts of the input simultaneously and figure out what's related to what.
    
    Image: A picture with a person and a dog    
    Self-attention helps: "The collar belongs to the dog, not the person"

    Overview: What This Code Does
        Input sequence (e.g., words) 
            ↓
        Convert to Query, Key, Value
            ↓
        Calculate attention weights (who looks at whom)
            ↓
        Weighted combination of values
            ↓
        Output sequence (with context from other words)

"""
class  SelfAttention(nn.Module):
  
    #d_embed: Embedding dimension；Larger = more capacity to store 
    # in_proj_bias and out_proj_bias: Whether to use bias in linear layers；Usually True (adds flexibility)
     
    def __init__(self, n_heads: int, d_embed: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        
        """
        self.in_proj: Projects input into Query, Key, Value all at once

        Why `3*d_embed?
        
        Input: d_embed dimensions
        Output: 3*d_embed dimensions
                ↓
        Split into 3 parts:
        - Query: d_embed dimensions
        - Key: d_embed dimensions  
        - Value: d_embed dimensions
        
        self.out_proj: Final transformation after attention
        Takes attention output back to original dimension
        """
        self.in_proj =nn.Linear(d_embed, 3*d_embed, bias=in_proj_bias)
        self.out_proj =nn.Linear(d_embed, d_embed, bias = out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads
    
    def forward(self, x: torch.Tensor, causal_mask =False):
        
        # x:(Batch_Size, Seq_Len, Dim)
        input_shape=x.shape
        batch_size, sequence_length, d_embed =input_shape
        
        # d_embed = n_heads * d_head
        intermim_shape =(batch_size, sequence_length, self.n_heads, self.d_head)
        
        #(Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_len, Dim*3) -> 3tensors of shape(Batch_Size, Seq_Len, Dim)
        """
        torch.chunk(tensor, chunks, dim): Split tensor into equal parts
        chunks=3: Split into 3 pieces
        dim=-1: Split along last dimension (the 1536)
        
        **Visual**:
        ```
        Combined (1536 dims):
        [QQQQQQQQQQ|KKKKKKKKKK|VVVVVVVVVV]
        512 dims   512 dims   512 dims
            ↓          ↓          ↓
            q          k          v
        ```
        
        ### What are Query, Key, Value?

        **Analogy**: Like a library search system

        **Query (Q)**: "What am I looking for?"
        - Example: Word "it" asks: "Who am I referring to?"

        **Key (K)**: "What do I contain?"
        - Example: Word "mat" says: "I'm a mat, I'm comfortable"

        **Value (V)**: "What information should I give?"
        - Example: Word "mat" provides: meaning, context, features
        """
        q,k,v =self.in_proj(x).chunk(3,dim=-1)
        
        #(Batch_Size，Seq_len, Dim) ->(Batch_Size, Seq_Len, H, Dim/H) ->(Batch_Size, H, Seq_Len, Dim/H)
        #tensor.view(shape): Reshape tensor without changing data
        """
        **Why transpose?**
        (if we don't transpose, when it comes to computing attention weights, (compare each query vector with all key vectors for that head:)
        the shape of Q*K^T would be head * head rather than seq * seq, 
        BUT Because attention’s whole purpose is to let each token attend to all other tokens , that is seq * seq; 
        RATHER THAN  let each head attend to all other heads within one token in a sequence, that is head*head )
        
        We want to process each head in parallel and independently:
        before, one seq dimension includes all head dimensions 
        BUT we hope one head dimension includes all seq dimensions so process each head independently

        
        ```
        Before: (Batch, Seq, Heads, Dim_per_head)
        Need to process: All sequences for Head1, then Head2, etc.

        After: (Batch, Heads, Seq, Dim_per_head)
        Easy to process: Head1 gets all sequences, Head2 gets all sequences, etc.
       
        q: (2, 8, 10, 64)
        k: (2, 8, 10, 64)
        v: (2, 8, 10, 64)

        Meaning:
        - 2 batches
        - 8 attention heads
        - 10 tokens in sequence
        - 64 dimensions per head
        """
        q=q.view(intermim_shape).transpose(1,2)
        k=k.view(intermim_shape).transpose(1,2)
        v=v.view(intermim_shape).transpose(1,2)
        
        #(Batch_Size, H, Seq_Len, Seq_Len)
        weight =q@k.transpose(-1,-2)
        """
        weight =q@k.transpose(-1,-2)
        ### What Does This Compute?

        **Attention scores between every pair of tokens!**

        **Result shape**: `(Batch, Heads, Seq_Len, Seq_Len)`
        - `(2, 8, 10, 10)` in our example

        **Meaning**:
        ```
        weight[0, 0, :, :] = Attention scores for Batch 0, Head 0
                            10×10 matrix

                    Token0  Token1  Token2  ...  Token9
        Token0  [     0.5     0.2     0.1    ...   0.05   ]
        Token1  [     0.1     0.6     0.15   ...   0.02   ]
        Token2  [     0.3     0.1     0.4    ...   0.08   ]
        ...
        Token9  [     0.05    0.02    0.1    ...   0.7    ]

        Each row: How much Token_i attends to all other tokens
        ```

        **Example**:
        ```
        Sentence: "The cat sat on the mat"
        Token2 ("sat") attention weights:
        - "The": 0.1  (low - not very related)
        - "cat": 0.6  (high - the cat is doing the sitting!)
        - "sat": 0.05 (low - self attention often low)
        - "on": 0.15  (medium - preposition is relevant)
        - "the": 0.05 (low)
        - "mat": 0.05 (low - "sat" doesn't attend much to destination)
        """
        
        if causal_mask:#Token at position i can only see tokens at positions 0 to i (not future tokens)
            #Mask where the upper triangle (above the principal diagonal) is made up of 1
            #ones_like:Returns a tensor filled with the scalar value 1, with the same size as input
            mask =torch.ones_like(weight, dtype=torch.bool).triu(1)#.triu(1): Upper triangle, diagonal offset by 1
            weight.masked_fill_(mask, -torch.inf)
            """
            1 .triu(1)
            Example 5×5 matrix:

            triu(0) - includes diagonal:    triu(1) - excludes diagonal:
            [1 1 1 1 1]                     [0 1 1 1 1]
            [0 1 1 1 1]                     [0 0 1 1 1]
            [0 0 1 1 1]                     [0 0 0 1 1]
            [0 0 0 1 1]                     [0 0 0 0 1]
            [0 0 0 0 1]                     [0 0 0 0 0]
            
            2. tensor.masked_fill_(mask, value): Fill positions where mask=True with value
            _ means in-place operation (modifies original tensor)
            
            3.Why -inf?
            After softmax, -inf becomes 0:softmax([-inf, 0, 1]) = [0, 0.27, 0.73]
                                    ↑ -inf becomes 0!
            ```

            So masked positions get zero attention weight!

            """
        """
        Before scaling:
        weight = [40, 32, 24, 16]  (large values!)

        After scaling:
        weight = [5, 4, 3, 2]  (smaller values)
        
        **Reason**: Dot product grows with dimension
        - With 64 dims: dot products can be huge!
        - Huge values → softmax saturation → vanishing gradients
        - Scaling prevents this

        **Formula**:
        ```
        attention = softmax(Q·K^T / √d_k)
                                ↑
                            Scaling factor
        """
        weight /= math.sqrt(self.d_head)
        weight =F.softmax(weight, dim=-1)
        """
        **`F.softmax(tensor, dim)`**: Convert scores to probabilities
        - `dim=-1`: Apply along last dimension (across all tokens being attended to)

        **What softmax does**:
        ```
        Input:  [2.0, 1.0, 0.5, 0.1]
                ↓ softmax
        Output: [0.51, 0.19, 0.11, 0.07]  (sum = 1.0)
        ```

        **Math**:
        ```
        softmax(x_i) = exp(x_i) / Σ exp(x_j)
        """
        
        #(Batch_Size, H, Seq_Len, Seq_Len) @ (Batch_Size, H, Seq_Len, Dim/H) -> (Batch_Size, H, Seq_Len, Dim/H)
        output =weight@v
        
        #(Batch_Size, H, Seq_Len, Dim/H)->(Batch_Size, Seq_Len, H, Dim/H)
        output=output.transpose(1,2)
        
        output=output.reshape(input_shape) # Merges 8 heads × 64 dims back into 512 dims
        """
        Before reshape:
        Token0: Head1[64] Head2[64] ... Head8[64]

        After reshape:
        Token0: [512 combined dimensions]
        """
        
        output=self.out_proj(output) # Mixes information from all heads
        
        #(Bacth_Size, Seq_Len,Dim)
        return output 
        """  
        output=self.out_proj(output)
        Why needed?

        - Mixes information from all heads
        - Learns how to combine multi-head outputs
        - Adds final transformation for downstream tasks
        
        e.g token_output = [h1_0, h1_1, h1_2, h1_3, h2_0, h2_1, h2_2, h2_3]
        
        The first 4 numbers came from head 1,
        the last 4 from head 2.
        
        Problem: heads are still separate subspaces

        Right now, each token’s embedding is basically:

        “Head1’s info” + “Head2’s info” stacked side by side.

        They don’t yet interact or mix —
        each head’s output occupies its own slice of the vector.

        So the model needs a way to:
            •	Blend information across heads
            •	Reintegrate it into one unified embedding for downstream layers

        That’s where out_proj comes in.
        
        This defines a weight matrix W_out (d_emd, d_emd)

        For each token vector 
        it computes:
        y_i = x_i W_out^T + b
        
        means:
        y= w1h1+...wnhn + b

        So every output dimension is a weighted sum of all input dimensions —
        i.e., it can mix across all heads freely.
        """
# Cross-attention : Allows one sequence (query) to attend to another sequence (key-value)
class CrossAttention(nn.Module):
    """
    Parameters :
        - n_heads : Number of attention heads for multi-head attention
        - d_embed : Embedding dimension of the query sequence
        - d_cross : Embedding dimension of the key-value sequence
        - in_proj_bias : Whether to use bias in input projection layers
        - out_proj_bias : Whether to use bias in output projection layer
    """
    def __init__(self, n_heads: int, d_embed: int, d_cross: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        #Projects query input to query space.Wq with shape (d_embed, d_embed)
        self.q_proj =nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        #Projects key input from cross-sequence to query embedding space
        #Input : d_cross dimensions → Output : d_embed dimensions
        #Matrix : Wk with shape (d_cross, d_embed)
        self.k_proj =nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        #- Purpose : Projects value input from cross-sequence to query embedding space
        #- Input : d_cross dimensions → Output : d_embed dimensions
        #- Matrix : Wv with shape (d_cross, d_embed)
        self.v_proj =nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        
        #- Purpose : Final output projection after attention computation
        #- Dimensions : d_embed → d_embed
        self.out_proj= nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads =n_heads
        self.d_head= d_embed//n_heads

    def forward(self, x, y):
        # x:(latent):(Batch_Size, Seq_Len_Q, Dim_Q)
        # y: (context): (Batch_Size, Seq_Len_KV, Dim_KV) =(Batch_Size, 77,768)
        """
        - x : Query sequence (e.g., image latents)
        - y : Key-Value sequence (e.g., text embeddings with 77 tokens, 768 dims)
        """
        input_shape =x.shape #input_shape : Stores original query shape for later restoration
        #Unpacking : Extracts batch size, sequence length, and embedding dimension
        batch_size, sequence_length, d_embed =input_shape
        #interim_shape : Target shape for multi-head reshaping
        #-1 means "infer this dimension" (will be sequence_length)
        interim_shape =(batch_size, -1, self.n_heads, self.d_head)
        
        #Multiply query by Wq
        q=self.q_proj(x)
        k=self.k_proj(y)
        v=self.v_proj(y)  
        
        
        # - .view(interim_shape) : Reshapes to (Batch_Size, Seq_Len, n_heads, d_head)
        # - .transpose(1,2) : Swaps dimensions → (Batch_Size, n_heads, Seq_Len, d_head)
        # - Purpose : Separates different attention heads for parallel processing
        
        q= q.view(interim_shape).transpose(1,2)
        k= k.view(interim_shape).transpose(1,2)
        v= v.view(interim_shape).transpose(1,2)
        
        # - Operation : Matrix multiplication Q × K^T
        # - k.transpose(-1,-2) : Transposes last two dimensions
        # - From: (Batch_Size, n_heads, Seq_Len_KV, d_head)
        # - To: (Batch_Size, n_heads, d_head, Seq_Len_KV)
        # - Result : (Batch_Size, n_heads, Seq_Len_Q, Seq_Len_KV)
        # - Meaning : Attention scores between each query and key position
        weight = q@k.transpose(-1,-2)
        weight /= math.sqrt(self.d_head)#Scaling factor : 1/√(d_head);Prevents attention scores from becoming too large
        weight=F.softmax(weight, dim=-1)#Applies softmax along the last dimension (Seq_Len_KV);Attention weights sum to 1 for each query position

        #Shape : (Batch_Size, n_heads, Seq_Len_Q, d_head)
        output = weight@v #Attention_weights × Values;Weighted sum of values
        #.contiguous() : Ensures memory layout is contiguous for the next view operation
        output = output.transpose(1,2).contiguous()
        
        #Reshapes back to original input shape;(Batch_Size, Seq_Len_Q, d_embed)
        output = output.view(input_shape)
        #Final linear transformation;Allows the model to learn how to combine the multi-head attention results
        output = self.out_proj(output)
        
        return output
    

    
    