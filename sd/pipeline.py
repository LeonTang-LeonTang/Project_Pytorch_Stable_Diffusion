
import torch
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler #from fileddpm.py； DDPM: Denoising Diffusion Probabilistic Model

# (all caps = constant convention)
WIDTH =512
HEIGHT =512
#VAE compresses images 8× in each dimension
"""
Image space:  512×512 pixels
Latent space: 64×64 (8× compression in each dimension)
Total compression: 64× (8×8)
"""
LATENTS_WIDTH= WIDTH//8
LATENTS_HEIGHT =HEIGHT//8

#output = w*(output_conditionaed - output_unconditioned) + output_unconditioned
# A weight indicates how much we want the model to pay attention to the conditioning signal(prompt)

def generate(
    prompt: str, # Type hint (string)
    uncond_prompt:str, # Negative prompt or empty string
    input_image=None,
    #strength: How strongly to modify the input image (1.0 = fully random noise, 0.0 = keep original)
    strength=0.8,
    #do_cfg — Whether to use “classifier-free guidance” (CFG) — the technique that blends conditioned and unconditioned predictions.
    do_cfg=True,
    cfg_scale=7.5,#How strongly to follow the text prompt (larger → more faithful but less diverse).
    sampler_name="ddpm",
    n_inference_steps=50,#Number of denoising steps，More steps = better quality, slower
    models={},#Empty dictionary，Will contain: encoder, decoder, diffusion, CLIP
    seed=None,
    device=None,
    idle_device=None,
    tokenizer=None,  
):
    with torch.no_grad():#Inference only (not training)
        if not (0< strength<=1):
            raise ValueError("strength must be between 0 and 1")
        
        #Function to move models to CPU when not in use (save GPU memory)
        #If idle_device exists, move model there when not used.
	    # Otherwise, do nothing.
        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x
        
        #torch.Generator: Random number generator
        #device=device: Which device (GPU/CPU)
        #Creates generator for reproducible randomness
        generator = torch.Generator(device=device)
        
        if seed is None:#No seed provided, .seed():Auto-generates random seed; Different results each run
            generate.seed()
        else: #.manual_seed(seed): Set specific seed,Same seed = same results (reproducible)
            generator.manual_seed(seed)
        
        clip = models["clip"] #Gets CLIP model from dictionary; e.g.models = {'clip': clip_model, 'encoder': vae_encoder}
        clip.to(device) #.to(device): Move model to device; device: GPU or CPU
        clip.eval()  # Set to evaluation mode for inference

        
        if do_cfg: #one for prompt , another one for uncond_prompt
            
                #.batch_encode_plus: Method that tokenizes text
                #[prompt]: List with one item；Why list? Method expects batch (multiple texts)
                #Pads shorter sequences to max length
                #77 = max token length CLIP expects
            #Tokenizer turns text → integer IDs (input_ids)
            # Convert the prompt into tokens using the tokenizer
            # truncation=True ensures prompts longer than 77 tokens are automatically cut
            cond_tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77, truncation=True).input_ids

            #(Batch_Size, Seq_Len)
            #Convert to Tensor
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)#torch.long: Data type (64-bit integer)
                
                #clip(cond_tokens): Pass tokens through CLIP
                # Input: (1, 77) - token IDs
                # Output: (1, 77, 768) - embeddings
                # Each token → 768-dim vector 
            #(Batch_Size, Seq_Len) ->(Batch_Size, Seq_Len, Dim)
            cond_context =clip(cond_tokens)
            
            #Tokenize Negative Prompt
                # Same process for negative/unconditional prompt:
                # Tokenize unconditioned prompt
                # Convert to tensor
                # Encode with CLIP
                # Result: uncond_context (1, 77, 768)
            uncond_tokens = tokenizer.batch_encode_plus([uncond_prompt],  padding="max_length", max_length=77, truncation=True).input_ids
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device = device)
            #(Batch_Size, Seq_Len)->(Batch_Size, Seq_Len， Dim)
            uncond_context=clip(uncond_tokens)
            # (Batch_Size, Seq_Len, Dim) + (Batch_Size, Seq_Len, Dim) -> (2 * Batch_Size, Seq_Len, Dim)
            context = torch.cat([cond_context, uncond_context])

        else:# No CFG
            # When CFG disabled:
            #     Only encode the main prompt
            #     No need for negative prompt
            #     Simpler, faster (but lower quality)
            #convert it into a list of tokens
            tokens =tokenizer.batch_encode_plus([prompt], padding='max_length', max_length=77, truncation=True).input_ids
            tokens =torch.tensor(tokens, dtype=torch.long, device=device)
            
            #(1,77,768)
            context =clip(tokens)
            
        to_idle(clip) #MOVE CLIP OFF GPU,to CPU (if idle_device set)Or does nothing (if no idle_device);Free up GPU memory
        
        if sampler_name =="ddpm":
            #Initialize a DDPM sampler.Pass generator for reproducible noise
            sampler =DDPMSampler(generator)
            #Configure number of steps
            sampler.set_inference_timesteps(n_inference_steps)
        else:
            raise ValueError('f"Unknown sampler(sampler_name)')
        
        #Define Latent Shape
        # 1: Batch size
        # 4: Latent channels (VAE uses 4)
        # LATENTS_HEIGHT: 64 (512 // 8)
        # LATENTS_WIDTH: 64 (512 // 8
        latents_shape =(1,4, LATENTS_HEIGHT, LATENTS_WIDTH)
        
        if input_image:#Conditional: Image-to-Image
            #Get VAE encoder from models dictionary and move to GPU
            encoder =models['encoder']
            encoder.to(device)
            encoder.eval()  # Set to evaluation mode for inference
            
            #Resizes image to 512×512
            input_image_tensor =input_image.resize((WIDTH, HEIGHT))
            #Converts PIL image to numpy array,Shape: (512, 512, 3);Format: Height, Width, Channel (HWC)
            input_image_tensor =np.array(input_image_tensor)
            
            #(Height, Width, Channel)
            #Convert numpy → torch
            input_image_tensor = torch.tensor(input_image_tensor, dtype= torch.float32)
            #Normalize pixel values:From: [0, 255] (standard image range) To: [-1, 1] (neural network range)
            """
            Why -1 to 1?

                Neural networks train better with centered data
                Mean ≈ 0 helps optimization
            """
            input_image_tensor =rescale(input_image_tensor, (0,255), (-1,1))
            
            
            # (Height, Width, Channel) ->(Batch_Size, Height, Width, Channel)
            #Add Batch Dimension at position 0; Why needed?: Models expect batches, even single images
            input_image_tensor = input_image_tensor.unsqueeze(0)
            
            #Permute to Channel-First; Why?: PyTorch convention is channel-first (NCHW format)
            # (Batch_Size, Height, Width, Channel) ->(Batch_Size, Channel, Height, Width)
            input_image_tensor =input_image_tensor.permute(0,3,1,2)
            
            
            #Generate Encoder Noise
            """  
            Why noise for encoder?

                VAE is probabilistic
                Adds slight randomness to encoding
                Helps with diversity  
                
            torch.randn(): Random normal distribution

                -latents_shape: (1, 4, 64, 64)
                -generator=generator: Use our seeded generator
                -device=device: Create on GPU/CPU
                -Values: Mean=0, Std=1 (standard normal)
            """
            encoder_noise = torch.randn(latents_shape, generator=generator, device=device)
            
            #Encode Image to Latent Space
            """
            Encode image:
                Input: (1, 3, 512, 512) image
                Output: (1, 4, 64, 64) latent
                Compresses 8× in each dimension
            """
            #run the image through the encoder of the VAE
            latents= encoder(input_image_tensor, encoder_noise)
            
            #Configure how much to denoise；Higher strength = more change to image
            sampler.set_strength(strength=strength)
            #Add initial noise  to encoded latents; sampler.timesteps[0]: First timestep
            latents =sampler.add_noise(latents, sampler.timesteps[0])
            #Why add noise?
                # Image-to-image needs starting point
                # Some noise gives room for changes
                # More noise = more dramatic changes
            to_idle(encoder) #Free up GPU memory (done with encoder)
            
        else:
            # if we are doing text-to-image, start with ranndom noise N(0,I)
            """
            Pure random noise:
                - Start from complete noise
                - No input image
                - Will denoise into generated image
                - Shape: (1, 4, 64, 64)
            """
            latents =torch.randn(latents_shape, generator=generator, device=device)
        
        #Load Diffusion Model
        diffusion = models['diffusion']
        diffusion.to(device)
        diffusion.eval()  # Set to evaluation mode for inference
        #Setup Progress Bar
        timesteps =tqdm(sampler.timesteps) #sampler.timesteps: List of timesteps (e.g., [999, 980, 961, ...])
        
        #Main Denoising Loop
        for i, timestep in enumerate(timesteps):
            #Convert timestep number to embedding;
            # get_time_embedding(timestep): Sinusoidal encoding
            #(1,320)
            time_embedding =get_time_embedding(timestep).to(device)
            
            #Current noisy latents become input
            #(Batch_Size, 4, Latents_Height, Latents_Width)
            model_input = latents
            
            if do_cfg:
                """
                .repeat(2, 1, 1, 1): Repeat tensor
                First 2: Repeat batch dimension 2 times
                Other 1s: Don't repeat other dimensions
                """
             # (Batch_Size, 4, Latents_Height, Latents_Width) -> (2 * Batch_Size, 4, Latents_Height, Latents_Width)
                #Duplicate latents → one for conditioned, one for unconditioned.
                model_input =model_input.repeat(2,1,1,1)
                
                
            #model_output is the predicted noise by the UNET
            #Run Diffusion Model
            model_output = diffusion(model_input, context, time_embedding)
            """
            model_input: Noisy latents (1, 4, 64, 64) or (2, 4, 64, 64)
            context: Text embeddings from CLIP
            time_embedding: Current timestep info

            What U-Net predicts: The noise to remove, not the clean image!
            If CFG:

            Input: (2, 4, 64, 64) (duplicated)
            Output: (2, 4, 64, 64) (two predictions)

            First: Prediction with prompt guidance
            Second: Prediction without prompt
            """
            
            if do_cfg:
                # .chunk(2): Split into 2 equal parts
                # Default: splits along dimension 0 (batch)
                # (2, 4, 64, 64) → Two (1, 4, 64, 64) tensors
                # Tuple unpacking assigns to two variables
                output_cond, output_uncond= model_output.chunk(2)
                
                #CFG formula
                """
                Breaking it down:
                    # Difference between conditioned and unconditioned:
                    diff = output_cond - output_uncond

                    # Scale the difference:
                    scaled_diff = cfg_scale * diff

                    # Add back to unconditioned:
                    model_output = scaled_diff + output_uncond
                    
                What it does:

                    - output_cond: Noise prediction with prompt
                    - output_uncond: Noise prediction without prompt
                    - diff: How much prompt changes prediction
                    - Amplify difference by cfg_scale (e.g., 7.5)
                    - Makes generation follow prompt more strongly
                """
                
                model_output = cfg_scale * (output_cond -output_uncond)+output_uncond
            
            #Remove noise predicted by the UNET
            """
            Denoising step:

                - timestep: Current time
                - latents: Current noisy image
                - model_output: Predicted noise
                - Returns: Less noisy latents
                
            What happens inside:
                # Simplified:
                latents = latents - (some_factor * model_output)
                
            After all iterations: Latents go from pure noise → clean latent image    
                
            """
            latents = sampler.step(timestep, latents, model_output)
            
        to_idle(diffusion)
        
        #Decode Latents to Image （RGB space)
        """
        Decode to pixel space:
            - Input: (1, 4, 64, 64) latents
            - Output: (1, 3, 512, 512) image
            - 8× upscaling in each dimension
            - From compressed space to pixel space
        """
        decoder = models['decoder']
        decoder.to(device)
        decoder.eval()  # Set to evaluation mode for inference
        
        # CRITICAL FIX: Scale latents properly before VAE decoding
        # The latents need to be scaled by 1/0.18215 before decoding
        # This is already done in the decoder, but we need to ensure it's correct
        images = decoder(latents)
        to_idle(decoder)
        
        #Post-Process Image
        """
        Rescale pixel values:

            From: [-1, 1] (model output range)
            To: [0, 255] (standard image range)
            clamp=True: Ensure values stay in range
        """     
        images = rescale(images, (-1,1), (0,255), clamp=True)
        
        #Permute to HWC Format
        #(Batch_Size, Channel, Height, Width) -> (Batch_Size, Height,Width, Channel)
            # Rearrange for display:
            #     From: (1, 3, 512, 512) PyTorch format (NCHW)
            #     To: (1, 512, 512, 3) Image format (NHWC)
        images =images.permute(0,2,3,1)
        
        #Convert to NumPy
        # to("cpu"): Move to CPU
        # torch.uint8: Convert to 8-bit unsigned integers
        images =images.to("cpu",torch.uint8).numpy()
        
        # Return First Image
            # Array indexing:
            #     images: Shape (1, 512, 512, 3)
            #     images[0]: Get first item → (512, 512, 3)
            #     Returns single image (no batch dimension)
        return images[0]
    
def rescale(x, old_range, new_range, clamp=False):
    """
    Rescale Formula:
    
    x_new = ((x-old_min) /(old_max -old_min) ) *(new_max -new_min)+new_min
    
    Function definition with 4 parameters:
    
        x: Tensor to rescale
        old_range: Tuple (min, max) of current range
        new_range: Tuple (min, max) of target range
        clamp=False: Whether to force values within new range bounds
        
    e.g.
        # Convert from [-1, 1] to [0, 255] (like image pixel values)
        x = torch.tensor([0.5])
        result = rescale(x, old_range=(-1, 1), new_range=(0, 255))
        # Step by step:
        # x = 0.5 - (-1) = 1.5          # Shift to zero
        # x = 1.5 * (255-0)/(1-(-1)) = 1.5 * 127.5 = 191.25  # Scale
        # x = 191.25 + 0 = 191.25       # Shift to target
    """
    #Tuple unpacking : Extracts minimum and maximum values
    old_min, old_max =old_range
    new_min, new_max = new_range
    """
    x-= old_min
        - Shift to zero : Moves the old range to start at 0
        - Example : If x = 0.5 and old_min = -1 , then x = 0.5 - (-1) = 1.5
        - Effect : Range (-1, 1) becomes (0, 2)
    """
    x-= old_min
    """
    x*=(new_max -new_min)/ (old_max -old_min)
        - Scale : Adjusts the range size
        - Formula : new_range_size / old_range_size
        - Example : Converting (0, 2) to (0, 255) : multiply by 255/2 = 127.5
    """
    x*=(new_max -new_min)/ (old_max -old_min)
    """
    x+= new_min
        - Shift to target : Moves to the desired starting point
        - Example : If new_min = 0 , no change. If new_min = 10 , adds 10
    """
    x+= new_min
    
    if clamp:#Forces values within bounds; .clamp(min, max) : Clips values below min to min, above max to max
        x = x.clamp(new_min, new_max)
    return x

#creates sinusoidal position encodings for timesteps
#Converts a single timestep number into a rich embedding vector
#timestep - integer representing the current denoising step
def get_time_embedding(timestep):
    # simlar to Transformer , PE(pos,2i) =sin(pos/10000 ^2i/dmodel)
    #BUT Computes 10000^(-i/160) for each i
    
    """
    - Create frequency array : Generates different frequencies for encoding
    - torch.arange(start=0, end=60) : Creates [0, 1, 2, ..., 59] (60 values)
    - /160 : Normalizes to [0, 1/160, 2/160, ..., 59/160]
    - -torch.arange(...)/160 : Makes negative: [0, -1/160, -2/160, ...]
    - torch.pow(10000, ...) : Computes 10000^(-i/160) for each i
    - Result shape : (60,) - 60 different frequencies
    
    # freqs will be approximately:
    # [10000^0, 10000^(-1/160), 10000^(-2/160), ..., 10000^(-59/160)]
    # = [1.0, 0.9886, 0.9773, ..., 0.0251]
    """
    #(160,)
    freqs= torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32)/160)
    
    """
    - Broadcasting multiplication : Combines timestep with all frequencies
        - torch.tensor([timestep]) : Converts timestep to tensor, shape (1,)
        - [:,None] : Reshapes to (1,1) for broadcasting
        - freqs[None] : Reshapes freqs from (160,) to (1,160)
        - Multiplication : (1,1) * (1,160) = (1,160)
        - Result : Each frequency multiplied by the timestep
    """
    #(1,160)
    x = torch.tensor([timestep], dtype=torch.float32)[:,None]*freqs[None]
    
    #(1,320)
    """
    - Apply trigonometric functions : Creates sinusoidal patterns
        - torch.cos(x) : Cosine of each frequency×timestep, shape (1,160)
        - torch.sin(x) : Sine of each frequency×timestep, shape (1,160)
        - torch.cat([cos, sin], dim=-1) : Concatenates along last dimension
    """
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)