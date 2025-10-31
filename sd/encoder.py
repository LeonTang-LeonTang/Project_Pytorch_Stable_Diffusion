import torch
from torch import nn
from torch.nn import functional as F
#imports your custom blocks. These must be defined in decoder.py. The encoder calls them as layers.
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

#Sequential is a container that applies modules in order. By inheriting, you can iterate for module in self: in forward().

""""
x = torch.randn(1, 3, 512, 512)      # Random RGB image
noise = torch.randn(1, 4, 64, 64)    # Random noise
```

**Journey**:
```
(1, 3, 512, 512)      ← Input image
    ↓ Conv2d
(1, 128, 512, 512)    ← 128 features
    ↓ 2× ResidualBlock
(1, 128, 512, 512)    
    ↓ Conv2d stride=2
(1, 128, 256, 256)    ← Downsampled to 1/2
    ↓ 2× ResidualBlock
(1, 256, 256, 256)    ← More channels
    ↓ Conv2d stride=2
(1, 256, 128, 128)    ← Downsampled to 1/4
    ↓ 2× ResidualBlock + Conv2d stride=2
(1, 512, 64, 64)      ← Downsampled to 1/8
    ↓ 3× ResidualBlock + Attention + ResidualBlock
(1, 512, 64, 64)      ← Deep processing
    ↓ GroupNorm + SiLU
(1, 512, 64, 64)      ← Normalized
    ↓ Conv2d (512→8)
(1, 8, 64, 64)        ← Compressed!
    ↓ Conv2d (1×1)
(1, 8, 64, 64)        ← Final features
    ↓ Split
mean: (1, 4, 64, 64)
log_var: (1, 4, 64, 64)
    ↓ Reparameterization
(1, 4, 64, 64)        ← Latent code!
    ↓ Scale by 0.18215
(1, 4, 64, 64)        ← Final output ✓

"""
class VAE_Encoder(nn.Sequential):
    
    # convoluitonal layer formula: Output size = (Input size-Kernel+2Padding)//Stride +1
    # To keep output same size as input: padding = (kernel_size -1) //2 (for stride = 1)
    
    def __init__(self):
        super().__init__(
            # (Batch_Size，Channels[here 3, RGB], Height, Width) -> (Batch_Size, 128, Height, Width)
            # input channels 3 (RGB), output channels 128（128 kernels/featuer maps)
            
            # The encoder is a neural network responsible for mapping input data to a latent space. Unlike traditional autoencoders that produce a fixed point in the latent space, 
            # the encoder in a VAE outputs parameters of a probability distribution—typically the mean and variance of a Gaussian distribution. 
            # This allows the VAE to model data uncertainty and variability effectively.Another neural network called a decoder is used to reconstruct the original data from the latent space representation.
            # Given a sample from the latent space distribution, the decoder aims to generate an output that closely resembles the original input data. 
            # This process allows the VAE to create new data instances by sampling from the learned distribution.
            # The latent space is a lower-dimensional, continuous space where the input data is encoded.
            # The variational approach is a technique used to approximate complex probability distributions. 
            # In the context of VAEs, it involves approximating the true posterior distribution of latent variables given the data, which is often intractable. 
            # The VAE learns an approximate posterior distribution. The goal is to make this approximation as close as possible to the true posterior.

            
            #  torch.nn.Conv2d(
            #     in_channels,     # number of input channels (e.g., 3 for RGB)
            #     out_channels,    # number of filters = number of output channels
            #     kernel_size,     # size of each filter (e.g., 3, 5, or (3,5))
            #     stride=1,        # how much the filter moves each step
            #     padding=0,       # number of pixels padded on each side
            #     dilation=1,      # spacing between kernel elements (advanced)
            #     groups=1,        # connection pattern (rarely changed)
            #     bias=True,       # whether to include a bias term
            #     padding_mode='zeros'  # padding type
 
            # )
            
            # •	You have 16 kernels.
            # •	Each kernel looks at all 3 channels (each kernel has shape [3, 3, 3]).
            # •	Each kernel produces one output feature map (after summing across RGB).
            # •	So you get 16 output feature maps.
                           
            # If your input is shaped as:
            # [batch_size, in_channels, height, width]
            # then your output will be:
            # [batch_size, out_channels, output_height, output_width]
            
            # out_channels = number of filters = number of kernels = number of feature maps
            # Each kernel (or filter) learns to detect a different type of feature in the input — e.g. edges, corners, textures, patterns, etc.
            
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            
            
        #     - `3` - Input channels (RGB: Red, Green, Blue)
        #     - `128` - Output channels (creates 128 feature maps)
        #     - `kernel_size=3` - Uses 3×3 filters
        #     - `padding=1` - Adds 1 pixel border so size stays the same
            
        #     Input: (1, 3, 512, 512)    [1 image, 3 colors, 512×512 pixels]
        #     Output: (1, 128, 512, 512) [1 image, 128 features, 512×512 pixels]
            
        #     RGB Image (3 channels)  →  [Conv]  →  128 feature maps
        #     [R G B]                              [Edge detector, texture detector, ...]
        #     Why increase channels from 3 → 128?

        # This is the most important part 👇

        # When you do feature extraction, you want to go from simple raw pixel info (3 colors)
        # to many complex learned features like:
        #     •	edges of various directions
        #     •	corners
        #     •	textures
        #     •	color blobs
        #     •	small shapes, etc.

        # Each of the 128 filters is learning to detect a different feature pattern.

        # So going from 3 → 128 isn’t expanding the data redundancy,
        # it’s expanding representational richness.

        # You can think of it like this:
        # Input (3 channels) Output (128 channels)
        # R, G, B color info -> “edge at 45°”, “texture”, “blue object”, “shadow”, “face part”, … (128 such features)
            
            
            #Why two blocks?: Learn more complex features before downsampling
            #Residual blocks usually contain convs + skip connection (identity) to stabilize training.
            #(Batch_Size, 128, Height, Width) -> (Batch_Size, 128 Height, Width)
            VAE_ResidualBlock(128,128),
            
            #(Batch_Size, 128, Height, Width)-> (Batch_Size, 128, Height, Width)
            VAE_ResidualBlock(128,128),
            
            # **What is a Residual Block?**
            #     A special block that helps training deep networks. It does:
            #     ```
            #     Input → [Convolution + Activation] → Output
            #     ↓                                    ↑
            #     └────────── Skip Connection ─────────┘
                
            #     Final Output = Output + Input
            
        
            # downsample:
            #(Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height/2, Width /2)
            
            # A downsampling conv with stride 2. Because padding=0, the spatial size will shrink more if input dims are not divisible by 2. 
            # To make it robust for odd sizes, we do manual padding in forward() before applying such layers.
            
            # why 2 convolution layer?
              # 1st conv: learn rich features (BY increase output_channels(numbers of kernels))
              # 2nd conv: compress spatially   (BY set stride =2)
            
            #In CNN/VAEs, two kinds of convolutions:
                #Type            
                #Feature extraction  Learn what patterns exist                     stride=1    nn.Conv2d(3,128,kernel_size=3, stride=1)
                #Downsampling       Learn where patterns occur at a coarser scale  stride=2    nn.Conv2d(128,128,kernel_size=3, stride=2)
            
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
            
            # channel expand:
            #（Batch_Size, 128, Height/2, Width/2) -> (Batch_Size, 256, Height/2, Width/2)
            #Residual block that increases channels from 128 → 256 (usually via a 1×1 conv in the residual path or similar inside block).
            VAE_ResidualBlock(128,256),
            
            #（Batch_Size, 256, Height/2, Width/2) -> (Batch_Size, 256, Height/2, Width/2)
            VAE_ResidualBlock(256,256),
            
            # downsample to Height/4
            #(Batch_Size, 256, Height/2, Width/2) -> (Batch_Size, 256, Height/4, Width/4)
            nn.Conv2d(256,256, kernel_size=3, stride=2, padding=0),
           
            # channel expand to 512
            #(Batch_Size, 256, Height/4 , Width/4) -> (Batch_Size, 512, Height/4, Width/4)
            VAE_ResidualBlock(256, 512),
            
            #(Batch_Size, 512, Height/4 , Width/4) -> (Batch_Size, 512, Height/4, Width/4)
            VAE_ResidualBlock(512, 512),
            
            # downsample to Height/8
            #(Batch_Size, 512, Height/4, Width/4) -> (Batch_Size, 512, Height/8, Widht/8)
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),
            
            #More residual processing at low spatial resolution to learn deep features.
            # deep feature processing at lowest spatial resolution
            #(Batch_Size, 512, Height/8, Width/8) -> (Batch_Size, 512, Height/8, Width/8)
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),
            
            # attention block (global spatial info)
            #(Batch_Size, 512, Height/8, Width/8) -> (Batch_Size, 512, Height/8, Width/8)
            VAE_AttentionBlock(512),
            
            #(Batch_Size, 512, Height/8, Width/8) -> (Batch_Size, 512, Height/8, Width/8)
            VAE_ResidualBlock(512,512),
            
            # normalization + activation
            #(Batch_Size, 512, Height/8, Width/8) -> (Batch_Size, 512, Height/8, Width/8)
            #What is GroupNorm? Normalizes the features to have mean=0, std=1 (makes training stable)
            nn.GroupNorm(32, 512),#32 groups dividing 512 total channels (512/32 = 16 channels per group). 
            # **What is SiLU?** (Sigmoid Linear Unit) A non-linear activation function: `SiLU(x) = x * sigmoid(x)`
            nn.SiLU(), #Activation function (also known as Swish): x * sigmoid(x)
            
            
            #(Batch_Size, 512, Height/8, Width/8) -> (Batch_Size, 512, Height/8, Width/8)
            #**Big compression!**: `512 channels → 8 channels`
            # **Why 8?**: Will split into mean (4) + log_variance (4)    
            nn.Conv2d(512, 8, kernel_size=3, padding=1), 
            #(Batch_Size, 512, Height/8, Width/8) -> (Batch_Size, 512, Height/8, Width/8)
            #1×1 conv to mix channels; keeps spatial dims and number of channels. This is useful as a final linear projection across channels.
            #kernel_size=1: A pointwise convolution (processes each pixel independently)
            nn.Conv2d(8,8, kernel_size=1, padding=0),
        )
        
    def forward(self, x:torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # x: (Batch_Size, Channel, Height, Width)
        # noise: (Batch_Size, Out_Channels, Height/8, Width/8)
        """
        x - The input image (e.g., 1×3×512×512)
        noise - Random noise for the reparameterization trick (e.g., 1×4×64×64)
        
        the whole loop:
         ww   1.	Iterates over each layer in the model.
            2.	Checks if the layer downsamples the image (stride = (2,2)).
            3.	If yes, it pads the image’s right and bottom edges by 1 pixel.
            4.	Then applies the layer.

        """
        
        for module in self: #Loop through each layer #self here is a nn.Sequential object 
            #pecial handling for stride=2 layers**: Add asymmetric padding
            if getattr(module, "stride", None) ==(2,2):
                #getattr(object, attribute_name, default_value)
                #Try to get the stride attribute of this module.
            	#If it doesn’t exist (like for ReLU, which has no stride), then return None.
                """
                **Why asymmetric padding?**
                ```
                Without padding:           With (0,1,0,1) padding:
                512×512 → [stride 2] →     512×512 → pad → 513×513 → [stride 2] → 256×256 ✓
                        → 255×255 ✗                                                (exact half)
                ```
                
                stride=(2,2)

                    1.	Single integer (e.g., stride=2)
                    •	Applies the same stride to both height and width
                    •	Equivalent to stride=(2, 2) internally.
                    2.	Tuple of two integers (e.g., stride=(2, 3))
                    •	Different stride for height vs width
                """
                #(Padding_Left, Paddding_Right, Padding_Top, Padding_Bottom)
                x = F.pad(x,(0,1,0,1))
            x =module(x) # applies the current layer (module) to the input x.
            
        # (Batch_Size, 8, Height, Height/8, Widht/8)-> two tensors of shape(Batch_Size, 4, Height/8, Width/8)
        # torch.chunk splits the tensor into two equal chunks along channel axis. 
        # Given x has 8 channels, this yields two tensors each with 4 channels.
        """
        - `x` - The tensor to split (1×8×64×64)
        - `2` - Split into 2 pieces
        - `dim=1` - Split along the channel dimension
        """
        mean, log_variance = torch.chunk(x,2,dim=1)
        
        #(Batch_Size, 4, Height/8, Width/8) -> ( Batch_Size, 4, Height/8, Width/8)
        #Clamps all elements in input into the range [ min, max ]. Letting min_value and max_value be min and max, respectively, this returns:
        #Keeps log_variance within a safe numeric range to avoid exp() underflow/overflow. 
        # If logvar is too negative, variance becomes ~0; if too positive, variance becomes huge.
        # Clamp the log variance between -30 and 20，Keeps training stable
        """
        Original: [-50, -10, 5, 25, 100]Ò
        After clamp(-30, 20): [-30, -10, 5, 20, 20]
                            ↑ clipped    ↑ clipped
        """
        log_variance =torch.clamp(log_variance,-30,20) 
        
        #(Batch_Size, 4, Height/8, Width/8) -> ( Batch_Size, 4, Height/8, Width/8)
        variance = log_variance.exp()#variance = e^{log_variance}. Convert from log space to normal space
        #(Batch_Size, 4, Height/8, Width/8) -> ( Batch_Size, 4, Height/8, Width/8)
        stdev = variance.sqrt()#Get standard deviation
        
        # Z=N(0,1) ->N(mean, variance) = X?
        # X =mean +stdev*Z
        #This operation produces a sample from N(mean, variance)
        # while retaining differentiability wrt mean and log_variance.
        
  
    #     **What it does**:
    #     - `noise` ~ N(0, 1) - Standard normal distribution
    #     - `mean + stdev * noise` ~ N(mean, stdev) - Shifted and scaled distribution
        
    #     **Example with numbers**:
    #     ```
    #     mean = [0.5, -0.2, 1.1, 0.3]
    #     stdev = [0.8, 0.5, 0.3, 0.9]
    #     noise = [0.2, -1.5, 0.0, 0.5]  (random from N(0,1))

    #     x = [0.5, -0.2, 1.1, 0.3] + [0.8, 0.5, 0.3, 0.9] * [0.2, -1.5, 0.0, 0.5]
    #     = [0.5, -0.2, 1.1, 0.3] + [0.16, -0.75, 0.0, 0.45]
    #     = [0.66, -0.95, 1.1, 0.75]
    #     ```

    #     **Why this trick?**
    #     - Allows backpropagation through random sampling
    #     - Without it, can't train the network!

    #     **Visual**:
    #     ```
    #     Standard Normal N(0,1):    Transformed N(mean, std):
    #          |                          |
    #       ___|___                    ___|___
    #      /   |   \                  /   |   \
    #     /    |    \                /    |    \
    #    ______|______         __________|__________
    #         0                      mean
            
    #     ← noise             ← stdev * noise + mean
    
        x =mean +stdev * noise
        
        # Scale the output by a constant
        
        #A constant scaling factor used by Stable Diffusion latent VAEs (empirical). 
        # It rescales latent magnitude to the range the diffusion model expects.
        
        # **Why multiply by 0.18215?**
        # - This is a magic constant from Stable Diffusion training
        # - Normalizes the latent space to have a good scale
        # - Makes training more stable
        
        x*= 0.18215
       
        return x
