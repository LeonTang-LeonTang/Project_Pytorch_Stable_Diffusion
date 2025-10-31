import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention, CrossAttention

"""
What is This U-Net For?
Purpose: The denoising network in a diffusion model

Input:  Noisy image + Text prompt + Timestep
         ↓
      [U-Net]
         ↓
Output: Predicted noise to remove

This U-Net is conditioned on:

Time (which denoising step we're at)
Text (what image to generate)
"""
class TimeEmbedding(nn.Module):
    """
    ### What is Time in Diffusion?

    **Diffusion generates images in steps**:
    ```
    Step 1000: ▓▓▓▓▓▓▓▓ (pure noise)
    Step 800:  ▓▓▓▒▒▓▓▓ (slightly denoised)
    Step 600:  ▓▒▒░░▒▓▓ (more structure)
    Step 400:  ▒░░  ░▒▒ (image emerging)
    Step 200:  ░     ░  (almost clear)
    Step 0:    [clean image]
    ```

    **The model needs to know**: "What step am I at?"
    - At step 1000: Remove a lot of noise
    - At step 200: Remove just a tiny bit of noise

    """
    
    def __init__(self, n_embd:int):
        super().__init__()
        #`n_embd`**: Input time embedding dimension
        """
        Reasons for not continuing expansion in the second Linear:
        - Memory efficiency : Continuing to expand (e.g., 1280 → 5120) would be computationally expensive
        - Information bottleneck : The 4x expansion provides enough capacity for time embedding
        - Standard practice : Most transformer architectures use 4x expansion in FFN layers
        - Diminishing returns : Beyond 4x, the benefits don't justify the computational cost
        """
        self.linear_1 = nn.Linear(n_embd, 4*n_embd)  # 320 → 1280 (expand)
        self.linear_2 = nn.Linear(4*n_embd, 4* n_embd)  # 1280 → 1280 (maintain)
        
    def forward(self, x:torch.Tensor)-> torch.Tensor:
        #Why two linear layers?

            # One layer: Simple transformation
            # Two layers: More expressive, can learn complex time representations
            # SiLU in between: Non-linearity
        # x:(1,320)
        
        x =self.linear_1(x) # (1, 320) → (1, 1280)
        
        x =F.silu(x)  # Apply SiLU activation
        
        x=self.linear_2(x) # (1, 1280) → (1, 1280)
        # (1,1280)
        return x

class UNET_ResidualBlock(nn.Module):
    
    """
    Complete flow:
    # Stage 1: Process spatial features(# Extract spatial features)
    # Stage 2: Add time conditioning ( # Combine spatial + temporal)
    # Stage 3: Refine combined features(# Refine combination)
    # Stage 4: Residual connection(# Add skip connection)
    
    ### Key Innovation: Time Injection

        **This is different from standard ResBlock!**

        **Standard ResBlock**:
        ```
        Input → [Conv] → [Conv] → Output
        ↓                         ↑
        └────────────────────────┘ (just add)
        ```

        **Diffusion ResBlock** (this one):
        ```
        Input → [Conv] → + Time → [Conv] → Output
        ↓             ↑                    ↑
        |    Time embedding added here    |
        └───────────────────────────────────┘ (residual)
    """
    
    def __init__(self, in_channels:int, out_channels: int, n_time=1280):
        super().__init__()
        
        #GroupNorm normalizes channels grouped into 32 groups
        self.groupnorm_feature = nn.GroupNorm(32, in_channels) #in_channels: Total channels to normalize
        #change channels (in_channels → out_channels);3×3 convolution; padding=1 preserves height/width.
        self.conv_feature=nn.Conv2d(in_channels, out_channels, kernel_size=3, padding =1)
        
        #Convert time embedding to match feature channels
        self.linear_time = nn.Linear(n_time, out_channels)
        """
        groupnorm_merged acts after we’ve added the time embedding.
        
        Why separate normalization and convolution:
            - Two-stage processing : First stage processes input features, second stage processes time-conditioned features
            - Different purposes :
                - groupnorm_feature + conv_feature : Initial feature extraction
                - groupnorm_merged + conv_merged : Refine features after time conditioning
            - Stability : Each stage needs its own normalization for training stability
            - Feature refinement : The second conv allows the model to learn how to combine spatial and temporal information
        """
        self.groupnorm_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        """
        self.residual_layer(residue): Apply residual transformation
        If channels match: Identity (no change)
        If different: 1×1 conv to match
        """
        if in_channels == out_channels:#if channels match.
            self.residual_layer = nn.Identity() #Simply returns input unchanged; e.g. input x and return x
        else:#1×1 convolution to change channels
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
    
    def forward(self, feature, time):
        #feature: Image features tensor
        #time: Time embedding tensor
        
        # feature:(Batch_Size, In_Channels, Height,Width)
        # time(1, 1280)
    
        residue = feature
        """
        Standard pattern: Normalize → Activate → Convolve
            - Normalization first : Stabilizes gradients and training
            - Activation : SiLU (Swish) provides smooth, non-linear transformation
            - SiLU advantages : Better gradient flow than ReLU, especially for deep networks
            - Proven effective : This order works best empirically in diffusion models
        """
        feature =self.groupnorm_feature(feature) # Normalize
        feature= F.silu(feature) # Activate
        feature =self.conv_feature(feature) # Convolve: change channels (in_channels → out_channels)
        
        time=F.silu(time)
        time= self.linear_time(time) # Project time to match channels
        
        # add two dimension to time to match feature dimension for merging
        merged =feature + time.unsqueeze(-1).unsqueeze(-1)
        
        #Standard pattern: Normalize → Activate → Convolve
        merged= self.groupnorm_merged(merged)
        merged= F.silu(merged)
        merged= self.conv_merged(merged)        
        """
        Residual connection logic:

            - residue : Original input (before any processing)
            - merged : Processed features (after time conditioning)
            - self.residual_layer(residue) : Adjusts original input to match output dimensions
        Why not alternatives:
            - merged + self.residual_layer(merged) : Would add processed features twice
            - residue + self.residual_layer(residue) : Would ignore all the processing work
            - Correct approach : Add original input (possibly channel-adjusted) to processed output
        """
        return merged + self.residual_layer(residue)
    
#When to use nn.Conv2d vs nn.Linear
#nn.Conv2d for:- Spatial data : Images, feature maps with Height×Width dimensions;- Local patterns : Exploits spatial locality with kernels
#nn.Linear for:- Sequential/vector data : Time embeddings, flattened features;- Global transformations : Mixing all input dimensions

class UNET_AttentionBlock(nn.Module):
    #d_context=768: Context dimension (from CLIP), default value
    def __init__(self, n_head:int, n_embed: int, d_context=768):
        super().__init__()
        channels = n_head* n_embed
        
        self.groupnorm =nn.GroupNorm(32, channels, eps=1e-6) #epsilon:Prevents division by zero in normalization
        
        # 1×1 convolution (pointwise)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        
        #self attention + cross attention + feed-forward linear
        #Why three layernorms? Different statistics : Each position has different input distributions;- Independent learning : Each layer learns its own normalization parameters

        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 =SelfAttention(n_head, channels, in_proj_bias=False)
        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 =CrossAttention(n_head, channels,d_context, in_proj_bias=False)
        self.layernorm_3=nn.LayerNorm(channels)
        
        self.linear_geglu_1 =nn.Linear(channels, 4* channels*2)
        #projects FF back to channels.
        self.linear_geglu_2 =nn.Linear(4*channels, channels) 
        
        #final 1×1 conv to mix channels and prepare to add the long skip.
        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
    
        
    def forward(self, x, context):
        
        # x: Image features
        #context: Text embeddings from CLIP
        
        # x: (Batch_Size, Features, Height, Width)
        # Context:(Batch_Size, Seq_Len, Dim)
        
        residue_long = x
        x= self.groupnorm(x)
        
        x= self.conv_input(x)
        
        n,c,h,w = x.shape #c:channels
        
        # (Batch_Size, Features, Height, Width) ->  (Batch_Size, Features, Height*Width)
        x= x.view((n,c,h*w))
        
        #(Batch_Size, Features, Height*Width)-> (Batch_Size, Height*Width,Features)
        x = x.transpose(-1,-2)
        
        # Normalization +Self Attention with skip connection
        
        residue_short = x
        
        x= self.layernorm_1(x)
        #self attention
        x = self.attention_1(x)
        x+= residue_short
        
        residue_short =x
        
        #Normalization +Cross Attention with skip connection
        x =self.layernorm_2(x)
        #Cross Attention
        x = self.attention_2(x, context)
        x += residue_short
        
        residue_short =x
        
        # Normalization +FF with GEGLU and skip connection
        
        x= self.layernorm_3(x)
        
        x, gate = self.linear_geglu_1(x).chunk(2,dim=-1)
        """
        Step 1: self.linear_geglu_1(x)
            Input: (2, 4096, 320)
            Output: (2, 4096, 2560) (because layer is 320 → 8×320)
        Step 2: .chunk(): Split tensor
            2: Into 2 pieces
            dim=-1: Along last dimension (2560)
            Returns tuple of 2 tensors: ((2, 4096, 1280), (2, 4096, 1280))
        Step 3: Unpack
            x = (2, 4096, 1280) (first half)
            gate = (2, 4096, 1280) (second half)
        """
        x =x* F.gelu(gate) #Apply GELU activation to gate
        x =self.linear_geglu_2(x) #Project back to original size
        x += residue_short
        
        #(Batch_Size, Height* Width, Features) ->(Batch_Size, Features, Height*Width)
        x=x.transpose(-1,-2)
        
        x=x.view((n,c,h,w))
        
        return self.conv_output(x)+residue_long

class Upsample(nn.Module):
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv =nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height*2, Width*2)
        x =F.interpolate(x, scale_factor=2, mode='nearest')
        """
        F.interpolate: PyTorch upsampling function
        scale_factor=2: Double each dimension
        mode='nearest': Nearest neighbor interpolation
        """
        return self.conv(x)
    
    
class SwitchSequential(nn.Sequential):
    def forward(self, x:torch.Tensor, context: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        for layer in self:
            # isinstance(object, class_or_tuple) returns True/False
            if isinstance(layer, UNET_AttentionBlock): # Check if layer is attention block
                x = layer(x,context)  # Attention needs context (text)
            elif isinstance(layer, UNET_ResidualBlock):
                x =layer(x, time)  # ResBlock needs time
            else:
                x =layer(x) # Regular layers (Conv, etc.)
        return x

class UNET(nn.Module):
    
    def __init__(self):
        super().__init__()
        """
        Key idea:
	•	Encoder gradually reduces spatial size (height/width)
	•	and increases feature channels (depth)
	•	so network learns coarse global features in deeper layers
        """
        
        # Conv2d(stride) -> ResidualBlock -> AttentionBlock
        # num_channels in ResidualBlock =  num_heads * dim_heads in AttentionBlock; 
        # So every stride =2, then channel will double and spatial dimensions halve.and num_heads , dim_heads in AttentionBlock would change correspondingly
        self.encoders = nn.ModuleList([
            
            #(Batch_Size, 4, Height/8, Width/8)
            SwitchSequential(nn.Conv2d(4,320, kernel_size=3, padding=1)),
            #Residual connection, maintains 320 channels;
            #Self-attention with 8 heads, 40 dimensions per head (320/8=40), that is why AttentionBlock(8,40)
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8,40)),
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8,40)),
            
            #Downsampling convolution with stride=2; Halves spatial dimensions
            #(Batch_Size, 320, Height/8, Width/8) ->(Batch_Size, 320, Height/16, Width/16)
            #Increases channels 320→640, attention with 8 heads × 80 dims
            SwitchSequential(nn.Conv2d(320,320, kernel_size=3,stride=2, padding=1)),
            SwitchSequential(UNET_ResidualBlock(320, 640), UNET_AttentionBlock(8,80)),
            SwitchSequential(UNET_ResidualBlock(640, 640), UNET_AttentionBlock(8,80)),
            
            #(Batch_Size, 640, Height/16, Width/16) ->(Batch_Size, 640, Height/32, Width/32)
            SwitchSequential(nn.Conv2d(640,640, kernel_size=3,stride=2, padding=1)),
            SwitchSequential(UNET_ResidualBlock(640, 1280), UNET_AttentionBlock(8,160)),
            SwitchSequential(UNET_ResidualBlock(1280,1280), UNET_AttentionBlock(8,160)),
            
            #(Batch_Size, 1280, Height/32, Width/32) ->(Batch_Size, 1280, Height/64, Width/64)
            SwitchSequential(nn.Conv2d(1280,1280, kernel_size=3,stride=2, padding=1)),
            #Deep processing : Two residual blocks at the deepest level;No attention : At this resolution, spatial attention is less beneficial
            SwitchSequential(UNET_ResidualBlock(1280,1280)),
            #(Batch_Size, 1280, Height/64, Width/64) ->(Batch_Size, 1280, Height/64, Width/64)                
            SwitchSequential(UNET_ResidualBlock(1280,1280))        
            
        ])
        
        # Purpose : Processes features at the deepest level (Height/64 × Width/64)
        # Structure : Residual → Attention → Residual
        self.bottleneck = SwitchSequential(
            # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            UNET_ResidualBlock(1280, 1280),
            # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            #  8(heads) * 160(dim per head) =1280(channels), so (8,160)
            UNET_AttentionBlock(8,160), 
            # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            UNET_ResidualBlock(1280, 1280),
        )
        
        # Upsample -> ResidualBlock -> AttentionBlock
        """
        Key idea:
            •	Decoder gradually upsamples back to original size
            •	At each step, it concatenates encoder features (skip connections)
        → that’s why input dims like 2560 (1280 from decoder + 1280 from encoder)
        
        Step                                                Operation                     
        UNET_ResidualBlock(2560,1280)                       Combine info from encoder and decoder
        Upsample(1280)                                      Double height and width
        UNET_AttentionBlock                                 Re-focus attention during reconstruction
        Channels gradually go down (1280 → 640 → 320 → 4)   Returns to 4-channel latent form


        """
        self.decoders = nn.ModuleList([
            #(Batch_Size, 2560, Height/64, Width/64) -> (Batch_Size, 1280, Height/64, Width/64)
            #- Input channels : 2560 = 1280 (current) + 1280 (skip connection from encoder)
            #- Skip connections : U-Net's key feature - concatenates encoder features
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),
            
        # (Batch_Size, 2560, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 32, Width / 32) 
            #Upsample(1280) : Increases spatial resolution while maintaining channels
            SwitchSequential(UNET_ResidualBlock(2560, 1280),Upsample(1280)),
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8,160)), # 8*160 = 1280
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8,160)),
            
        # (Batch_Size, 1920, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 16, Width / 16)
            #- Input : 1920 = 1280 + 640 (skip connection from encoder at this level)
            #Upsampling : Height/32 → Height/16
            # Channel reduction : 1920→640, then 1280→640
            # Attention : 8×80 (matching the encoder at this resolution)
            SwitchSequential(UNET_ResidualBlock(1920, 1280), UNET_AttentionBlock(8,160), Upsample(1280)),
            SwitchSequential(UNET_ResidualBlock(1920, 640), UNET_AttentionBlock(8,80)), # 8*80=640
            SwitchSequential(UNET_ResidualBlock(1280, 640), UNET_AttentionBlock(8,80)),
            #- Input : 960 = 640 + 320 (skip connection)
            #Upsampling : Height/16 → Height/8
        # (Batch_Size, 960, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 8, Width / 8)
            SwitchSequential(UNET_ResidualBlock(960, 640), UNET_AttentionBlock(8,80), Upsample(640)),
            SwitchSequential(UNET_ResidualBlock(960, 320), UNET_AttentionBlock(8,40)),
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8,40)),
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8,40)),
            
        ])

    def forward(self,x, context, time):
        # x:(Batch_Size, 4, Height/8, Width/8)
        # context: (Batch_Size, Seq_Len, Dim)
        # time:(1,1280)
        skip_connections=[]
        for layers in self.encoders:
            x =layers(x, context, time)
            skip_connections.append(x)

        x= self.bottleneck(x, context, time) #the middle layer of the network

        for layers in self.decoders:
            #Since we always concat with the skip connection of the encoder, the number of features increases before being sent to the decoder's layer
            #x = current decoder output
            #skip_connections.pop() = removes and returns last item from list
            x =torch.cat((x, skip_connections.pop()), dim=1)
            """  
                ## The Two Tensors Being Concatenated:
                
                ### **1st Tensor: `x`**
                - This is the **current output from the previous decoder layer** (or from the bottleneck in the first iteration)
                - It's been **upsampled** (made bigger/higher resolution)
                - Contains **high-level, abstract features**
                - Shape example: `(Batch_Size, 256, Height/4, Width/4)`
                
                ### **2nd Tensor: `skip_connections.pop()`**
                - `.pop()` removes and returns the **last item** from the `skip_connections` list
                - This is the **corresponding encoder output** from the same resolution level
                - Contains **low-level, detailed features** (edges, textures, fine details)
                - Shape example: `(Batch_Size, 256, Height/4, Width/4)`
                
                ---
                
                ## After Concatenation (`dim=1`):
                
                - `dim=1` means concatenate along the **channel dimension**
                - **Result shape**: `(Batch_Size, 512, Height/4, Width/4)`
                - Notice channels doubled: 256 + 256 = 512
                
                ---
                
                ## Visual Example:
                ```
                Encoder output (saved earlier): [Batch=1, Channels=256, H=64, W=64]
                Current x (from decoder):       [Batch=1, Channels=256, H=64, W=64]
                                                           ↓
                After torch.cat(..., dim=1):    [Batch=1, Channels=512, H=64, W=64]
            """
            x =layers(x, context, time)
        return x
            
class UNET_OutputLayer(nn.Module):#Final layer to convert features to output
    #Conv2d : Final convolution to desired output channels
    def __init__(self, in_channels:int, out_channels:int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv =nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        #Pipeline : GroupNorm → SiLU activation → Convolution
        
        # x:(Batch_Size, 320, Height/8, Width/8)
        
        x= self.groupnorm(x)
        
        x= F.silu(x)
        
        x=self.conv(x)
        
        #(Batch_Size, 4, Height/8, Width/8)
        return x

class Diffusion(nn.Module):
    
    def __init__(self):
        super().__init__()

        #TimeEmbedding : Processes timestep information
        self.time_embedding =TimeEmbedding(320)
        #UNET : Main denoising network
        self.unet =UNET()
        #UNET_OutputLayer : Converts to 4-channel output
        self.final = UNET_OutputLayer(320,4)

    def forward(self, latent:torch.Tensor, context:torch.Tensor, time:torch.Tensor):
        #latent: (Batch_Size, 4, Height/8, Width/8)
        #context: (Batch_Size, Seq_Len, Dim)
        #time:(1,320)
        
        """
        Forward Pass :

        1. Time embedding : Converts timestep to 1280-dimensional embedding
        2. U-Net processing : Main denoising with latent, context (text), and time
        3. Final layer : Converts 320 channels back to 4-channel latent space
        4. Output : Denoised latent representation
        """
        #(1,320) ->(1,1280)
        time =self.time_embedding(time)
        
        #(Batch, 4, Height/8, Width/8) ->(Batch, 320, Height/8, Width/8)
        output =self.unet(latent, context, time)
        
        #(Batch, 320, Height/8, Width/8) ->(Batch, 4, Height/8, Width/8)
        output =self.final(output)
        
        #(Batch, 4, Height/8, Width/8)
        return output
