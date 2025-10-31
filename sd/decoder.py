import torch 
from torch import nn
from torch.nn import functional as F #for operations like activation functions (F.silu), padding, etc.
from attention import  SelfAttention # custom SelfAttention class from attention.py

class VAE_AttentionBlock(nn.Module):
    
    def __init__(self, channels:int):
        super().__init__()
        #Creates a Group Normalization layer: divides the channels into 32 groups.
        self.groupnorm = nn.GroupNorm(32, channels) 
        self.attention =SelfAttention(1, channels) #instantiates the SelfAttention module with arguments
        
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        # x: (Batch_Size, Features, Height, Width)
        
        residue =x #Stores the input x in residue for later residual connection (skip connection) so the original information can be added back.
        
        n, c, h, w = x.shape #unpacks the dimensions of x: batch_size, number of channels, height, width
        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height * Width)
        x =x.view(n, c,h*w) #Reshapes x from shape (n, c, h, w) to (n, c, h*w).
        
        #(Batch_Size, Features, Height*Width) ->(Batch_Size, Height*Width, Features)
        x = x.transpose(-1,-2) #Swaps two dimensions of x.
        
        #(Batch_Size, Height*Width, Features) ->(Batch_Size, Height*Width, Features)
        x = self.attention(x) #Applies the SelfAttention module to x
        
        # (Batch_Size, Height*Width, Features) ->(Batch_Size, Features, Height*Width)
        x = x.transpose(-1,-2)#Swaps dimensions back: from (n, h*w, c) → (n, c, h*w).
        
        #(Batch_Size, Features, Height*Width) ->(Batch_Size, Features, Height, Width)
        x = x.view((n,c,h,w)) #	Reshapes back to original 4D shape: batch, channel, height, width.
        
        x+=residue #Adds the original input (residue) to the transformed x (skip connection)
        return x


class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        #First normalization: normalizes in_channels grouped into 32 groups.
        self.groupnorm_1 =nn.GroupNorm(32, in_channels)
        #First convolution: kernel 3×3, same padding so spatial dims unchanged, channels go from in_channels → out_channels.
        self.conv_1 =nn.Conv2d(in_channels, out_channels, kernel_size =3, padding=1)
        
        #Second normalization using output channels from conv_1.
        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        #Second convolution: keeps channels the same (out_channels → out_channels).
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        """
        ## Visual Comparison
            ### Case 1: Same Channels (128 → 128)
            ```
            Input: 128 channels          Output: 128 channels
                ║ ║ ║                       ║ ║ ║
                ║ ║ ║  → [Processing] →     ║ ║ ║
                ║ ║ ║                       ║ ║ ║
                ╚═╬═╝                       ║ ║ ║
                └─────── Identity ─────→  ║ ║ ║
                                            ╚═╩═╝
                                            +
                                            Final
            ```
            Just copy! No transformation needed.

            ### Case 2: Different Channels (128 → 256)
            ```
            Input: 128 channels              Output: 256 channels
                ║ ║                           ║ ║ ║ ║
                ║ ║  → [Processing] →         ║ ║ ║ ║
                ║ ║                           ║ ║ ║ ║
                ╚═╝                           ║ ║ ║ ║
                └─→ [1×1 Conv: 128→256] →   ║ ║ ║ ║
                    Expand to 256 channels   ╚═╩═╩═╝
                                                +
                                            Final
        """
        # Skip connection path; decides if we need to transform the input first.
        if in_channels ==out_channels:
            self.residual_layer = nn.Identity()# Same channels: just copy
        else:# Different channels: transform to match
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size =1, padding=0)
        """
        e.g.
        # Main path
        x = conv_1(x)  # (1, 128, 64, 64) → (1, 256, 64, 64)  ← Changed to 256!
        x = conv_2(x)  # (1, 256, 64, 64) → (1, 256, 64, 64)

        # Skip path
        residue = Conv2d_1x1(residue)  # (1, 128, 64, 64) → (1, 256, 64, 64)  ← Transform!

        # Combine
        output = x + residue  # (1, 256, 64, 64) + (1, 256, 64, 64) ✓
        """
        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x：（Batch_Size, In_channels, Height, Width)
        
        residue =x
        
        x= self.groupnorm_1(x)
        
        x = F.silu(x)
        
        x= self.conv_1(x)
        
        
        x=self.groupnorm_2(x)
        
        x = F.silu(x)
        
        x=self.conv_2(x)
        
        return x+self.residual_layer(residue) # Transform if needed
    
class VAE_Decoder(nn.Sequential):
    """
    What Does the Decoder Do? 
    Decompressing a ZIP file back to the original image!
        Compressed latent code (64×64×4)
        ↓ Expand and add details
        ↓ Upsample (make bigger)
        ↓ Add more details
        ↓ Upsample again
        ↓ Keep expanding
        Full image (512×512×3)
        ```

        **Journey**:
        ```
        Input:  (Batch, 4, 64, 64)      ← Compressed latent
        Output: (Batch, 3, 512, 512)    ← Full RGB image
        
    **`nn.Sequential`**: Layers execute one after another in order
    - No need to manually write forward pass for each layer
    - Data flows through automatically

    **Structure**:
    ```
    Layer 1 → Layer 2 → Layer 3 → ... → Output
    
        
    ## Complete Journey Visualization
    ```
    INPUT: Latent Code
    (1, 4, 64, 64)
        ↓ Scale back (÷ 0.18215)
        ↓ Conv 4→4
        ↓ Conv 4→512 (expand features)
    (1, 512, 64, 64)
        ↓ 5× ResBlock + Attention (deep processing)
    (1, 512, 64, 64)
        ↓ Upsample 2× → 128×128
        ↓ Conv + 3× ResBlock (refine)
    (1, 512, 128, 128)
        ↓ Upsample 2× → 256×256
        ↓ Conv + 3× ResBlock (512→256 channels)
    (1, 256, 256, 256)
        ↓ Upsample 2× → 512×512
        ↓ Conv + 3× ResBlock (256→128 channels)
    (1, 128, 512, 512)
        ↓ GroupNorm + SiLU
        ↓ Conv 128→3 (to RGB)
    OUTPUT: RGB Image
    (1, 3, 512, 512)
    ```

    ---

    ## Key Patterns in the Decoder

    ### Pattern 1: Gradual Upsampling
    ```
    64×64 → 128×128 → 256×256 → 512×512

    Each stage 2× bigger
    ```

    ### Pattern 2: Channel Reduction
    ```
    4 → 512 → 512 → 512 → 256 → 128 → 3

    Start high (for rich features)
    Gradually reduce (for efficiency)
    End at 3 (RGB output)
    ```

    ### Pattern 3: Processing After Upsample
    ```
    Upsample (blocky)
        ↓
    Conv (smooth)
        ↓
    ResBlocks (refine)
        ↓
    Ready for next upsample
    ```

    ---

    ## Comparing Encoder vs Decoder

    | Aspect | Encoder | Decoder |
    |--------|---------|---------|
    | **Direction** | Compress (big→small) | Expand (small→big) |
    | **Spatial size** | 512→256→128→64 (↓) | 64→128→256→512 (↑) |
    | **Channels** | 3→128→256→512→4 | 4→512→256→128→3 |
    | **Key operation** | Conv with stride=2 (downsample) | Upsample + Conv (upsample) |
    | **Output** | Latent code (4, 64, 64) | RGB image (3, 512, 512) |

    **They're mirror images!**
    ```
    Encoder:                    Decoder:
    512×512×3                   64×64×4
        ↓ Shrink                   ↑ Grow
    256×256×...                128×128×...
        ↓ Shrink                   ↑ Grow
    128×128×...                256×256×...
        ↓ Shrink                   ↑ Grow
    64×64×4                    512×512×3
    """
    def __init__(self):
        """
            Feature Mixing
            
            The 4 latent channels from the encoder might represent independent, abstract features:
            ```
            Channel 0: Encodes "brightness/lighting"
            Channel 1: Encodes "color information"
            Channel 2: Encodes "texture patterns"
            Channel 3: Encodes "spatial structure"
            
            # Before: 4 independent channels
            [brightness, color, texture, structure]

            # After: 4 mixed channels (learned combinations)
            [0.5*bright + 0.3*color, ...]

        """
            
        super().__init__(
            nn.Conv2d(4,4, kernel_size=1, padding=0), # mixes the 4 latent channels by Pointwise convolution and same channels
            
            #Need rich features (512 channels) to reconstruct details
            nn.Conv2d(4,512, kernel_size=3, padding=1),#`4 → 512`: Massive increase in features;padding=1`: Keeps spatial size same
            
            
            # **5 Residual Blocks + 1 Attention Block!**

            # **Why so many at this stage?**
            # - Smallest spatial resolution (64×64)
            # - Most compressed representation
            # - Need deep processing to understand high-level semantics
            # - Attention helps capture global relationships

            # **Shape**: Stays `(Batch, 512, 64, 64)` throughout

            # **What they do**:
            # ```
            # [Compressed code] 
            #     ↓ ResBlock: Learn features
            #     ↓ Attention: Look at relationships  
            #     ↓ ResBlock: Refine features
            #     ↓ ResBlock: More refinement
            #     ↓ ResBlock: Even more
            #     ↓ ResBlock: Final processing
            # [Rich feature map, ready to expand]
            
            # ### Why Multiple ResidualBlocks?

            # **Single layer is too shallow!**

            # **Think of it like painting**:
            # ```
            # 1 ResBlock: Rough sketch
            # 2 ResBlocks: Basic shapes  
            # 3 ResBlocks: More details
            # 4 ResBlocks: Fine details
            # 5 ResBlocks: Very refined features
            # ```

            # **Each ResBlock adds processing depth**:
            # ```
            # Layer 1: Detects basic patterns (edges, blobs)
            # Layer 2: Combines patterns (corners, simple shapes)
            # Layer 3: Forms objects (eyes, noses)
            # Layer 4: Refines objects (better proportions)
            # Layer 5: Final polish (smooth transitions)
            
            VAE_ResidualBlock(512,512),
            VAE_AttentionBlock(512),
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),
            #(Batch_Size, 512, Height/8, Width/8) -> (Batch_Size, 512, Height/8, Width/8)
            VAE_ResidualBlock(512,512),
            
        
            
# **Attention's role**:
# - After some local processing (first ResBlock)
# - Look globally across the entire image
# - Find long-range dependencies
# - Then continue refining with that global knowledge

# **Example**:
#     ```
#     Without attention:
#     - Top-left corner doesn't know about bottom-right
#     - Each region processed independently
#     - Might have inconsistencies

#     With attention (middle position):
#     - Build some features first (ResBlock)
#     - Then look globally (Attention)
#     - Use global info to refine (more ResBlocks)
#     - Result: Coherent, consistent image
#     ```

#     **Why not at the start?** 
#     - Attention on raw latents is less useful
#     - Better to build some features first

#     **Why not at the end?**
#     - Need to refine after global understanding
#     - Final ResBlocks incorporate the global context

#     ### Why NOT just one of each?

#     **Experiment results show**:
#     ```
#     1 ResBlock:  Poor reconstruction, blurry
#     3 ResBlocks: Better, but still missing details  
#     5 ResBlocks: Good reconstruction quality
#     7 ResBlocks: Diminishing returns (not much better, slower training)
            
              
            #(Batch_Size, 512, Height/8, Width/8) -> (Batch_Size, 512, Height/4, Width/4)
            nn.Upsample(scale_factor=2),#`nn.Upsample(scale_factor=2)`: Make image **2× bigger** in each dimension
            
        
        #     why use the Conv after upsample?
        #     **Smoothing and refinement**:
        #     ```
        #     After upsample (blocky):
        #     [1 1 2 2]
        #     [1 1 2 2]
        #     [3 3 4 4]
        #     [3 3 4 4]

        #     After 3×3 conv (smooth):
        #     [1.0 1.2 1.8 2.0]
        #     [1.5 1.5 2.5 2.5]  ← Smoother transitions!
        #     [2.5 2.5 3.5 3.5]
        #     [3.0 3.2 3.8 4.0]
            
        #      **Learned** interpolation (better than bilinear/bicubic)
        #     - Network learns best way to fill in details
            
        # How to Determine Number of ResidualBlocks?

        #     This is **empirical** (learned from experiments), but here are the principles:

        #     ### Principle 1: Resolution-Dependent
        #     ```
        #     At 64×64 (smallest):   5 ResBlocks   ← Most processing
        #     At 128×128:            3 ResBlocks   
        #     At 256×256:            3 ResBlocks
        #     At 512×512 (largest):  3 ResBlocks   ← Less processing
        #     ```

        #     **Why more at smaller resolution?**
        #     - Smallest resolution has most compressed info
        #     - Needs deepest understanding
        #     - Like focusing hard on reading small print

        #     **Why fewer at larger resolution?**
        #     - Already has most details from upsampling
        #     - Just needs refinement
        #     - Too much processing wastes computation
            
        #     ### Principle 2: Computational Budget

        #     **Memory usage** grows with resolution:
        #     ```
        #     64×64:   Can afford 5 ResBlocks (small memory footprint)
        #     512×512: Can only afford 3 ResBlocks (huge memory footprint)
        #     ```

        #     **Time consideration**:
        #     ```
        #     Total computation = Resolution × Number of layers

        #     64×64 × 5 blocks = 20,480 operations
        #     512×512 × 5 blocks = 1,310,720 operations ← 64× more!

        #     Using 3 blocks at large resolution: 786,432 operations (manageable)
        #     ```
            
            
            nn.Conv2d(512,512, kernel_size=3, padding=1),
            
            #Refine features at this scale
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),
        
            
            # Why One Upsample + One Conv + Three ResBlocks?
            
            # ### This is a Standard Pattern!

            # **Step-by-step reasoning**:

            # **1. Upsample**: 
            # ```
            # Purpose: Increase spatial resolution
            # Result: Blocky, needs refinement
            # ```

            # **2. One Conv**:
            # ```
            # Purpose: Immediate smoothing of upsampling artifacts
            # Why one? Initial cleanup is enough, deeper refinement comes next
            # ```

            # **3. Three ResBlocks**:
            # ```
            # Why three specifically?

            # One ResBlock:   Basic refinement, not enough depth
            # Two ResBlocks:  Better, but still could improve  
            # Three ResBlocks: Good balance - enough depth without over-processing ✓
            # Four ResBlocks:  Diminishing returns, slower, more memory
            # Five ResBlocks:  Overkill at this stage

            # ### Why This Specific Count?

            #     **It's based on empirical findings**:

            #     **ImageNet studies** (many papers):
            #     ```
            #     2 ResBlocks: Adequate for simple features
            #     3 ResBlocks: Better for complex features ← Sweet spot!
            #     4+ ResBlocks: Marginal gains, higher cost
            # **Stable Diffusion experiments**:

            # Testing showed 3 ResBlocks per stage gives best quality/speed trade-off
            
            #(Batch_Size, 512, Height/4, Width/4) ->(Batch_Size, 512, Height/2, Width/2)
            nn.Upsample(scale_factor=2),
            
            
            # Why Some Channels Change in ResidualBlocks, Others Don't?
            #     Change channels when:

            #     Transitioning between resolution stages
            #     Need to reduce computational cost
            #     Features become more concrete (less abstract)

            #     Keep channels when:

            #     Processing at same resolution
            #     Building depth without changing capacity


            
            nn.Conv2d(512,512, kernel_size=3, padding=1),
            
            
            # **Trade-off:**
            # ```
            # High resolution + High channels = Out of memory!
            # High resolution + Low channels = ✓ Fits in memory
            
            # ### Pattern Summary:
            # ```
            # Resolution  Channels  Why
            # ─────────────────────────────────────────────────────
            # 64×64       512       Bottleneck: needs max capacity
            # 128×128     512       Still abstract features
            # 256×256     256       Reduce: memory + more concrete features
            # 512×512     128       Reduce: memory + very concrete features
            # Output      3         RGB image
            # ```
            # Why reduce to 256 here?

            #     - Resolution doubled: 256×256 = 4× more pixels than 128×128
            #     - Memory concern: 512 channels × 256×256 = 33M values (expensive!)
            #     - Feature abstraction: At 256×256, features are more concrete (textures, edges), need less capacity
            #     - Computational efficiency: Halving channels = 4× less computation
            
            #Larger spatial size (256×256) = more memory;Don't need as many channels for finer details
            VAE_ResidualBlock(512,256),
            VAE_ResidualBlock(256,256),
            VAE_ResidualBlock(256,256),
            
            # (Batch_Size, 256, Height/2, Width/2) -> (Batch_Size, 256, Height, Width)
            nn.Upsample(scale_factor=2),
            
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            # **Why reduce to 128?**
            # 1. **Full resolution**: 512×512 = massive!
            # 2. **Simple features**: At full resolution, mostly fine details (edges, textures), don't need high capacity
            # 3. **Memory critical**: 256 × 512 × 512 = 67M values per image!
            # 4. **Final output is only 3 channels**: Gradually transitioning down
            
            VAE_ResidualBlock(256,128),
            VAE_ResidualBlock(128,128),
            VAE_ResidualBlock(128,128),
            
            nn.GroupNorm(32, 128), #Each group: 128/32 = 4 channels
            nn.SiLU(),#Non-linear activation;#SiLU(x) = x × sigmoid(x)
            
            #(Batch_Size, 128, Height, Width) -> (Batch_Size, 3, Height, Width)
            #128 channels → 3 channels (RGB);3 channels = Red, Green, Blue
            nn.Conv2d(128, 3,kernel_size=3, padding=1)
            
        )
#           Reason 2: Feature Hierarchy

#             **At different resolutions, features have different meanings:**

#             **Small resolution (64×64) with many channels (512)**:
#             ```
#             Abstract, semantic features:
#             - Channel 1: "Is there a face?"
#             - Channel 2: "Overall pose"
#             - Channel 3: "Lighting direction"
#             - ... (512 different abstract concepts)

#             Need MANY channels to encode abstract, compressed information
#             ```

#             **Large resolution (512×512) with few channels (128)**:
#             ```
#             Concrete, local features:
#             - Channel 1: "Horizontal edge at this pixel"
#             - Channel 2: "Red color intensity"
#             - Channel 3: "Texture roughness"
#             - ... (128 concrete, local features)

#             Need FEWER channels - information is already spatially distributed!
#             ```

#             **Analogy:**
#             ```
#             Small resolution = Map legend (needs many symbols)
#             Large resolution = Actual detailed map (symbols already placed in space)
#             ```

#             ### Reason 3: Information Capacity Balance

#             **Total information capacity** = Channels × Height × Width
#             ```
#             At 64×64:
#             512 × 64 × 64 = 2,097,152 total values

#             At 512×512:
#             128 × 512 × 512 = 33,554,432 total values ← 16× more!

#             Even with 4× fewer channels, still 16× more information at full resolution!

#             Computation = Channels² × Height × Width (approximately, for convolutions)
#             # At 64×64 with 512 channels:
#             512² × 64 × 64 = 1,073,741,824 operations

#             # At 512×512 with 512 channels:
#             512² × 512 × 512 = 68,719,476,736 operations ← 64× more!

#             # At 512×512 with 128 channels:
#             128² × 512 × 512 = 4,294,967,296 operations ← 16× less than above
#             ```

# **Without reducing channels, training would be impossibly slow!**

        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # x: (Batch_Size, 4, Height / 8, Width/8)
        
        x /= 0.18215
        
        # Remember the encoder? It multiplied by 0.18215 at the end.
        # Decoder reverses this: Divide by 0.18215
        # Why?

        # Encoder scaled down for training stability
        # Decoder scales back up to normal range

        
        
        for module in self:
            x =module(x)
        #(Batch_Size, 3, Height, Width)
        return x

