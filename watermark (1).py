import torch
from scipy.stats import norm,truncnorm
from functools import reduce
from scipy.special import betainc
import numpy as np
from Crypto.Cipher import ChaCha20
from Crypto.Random import get_random_bytes
from tqdm import  tqdm
import random
from itertools import combinations

class Gaussian_Shading_chacha:
    def __init__(self, ch_factor, hw_factor, fpr, user_number , keepuser=False):
        self.ch = ch_factor
        self.hw = hw_factor
        self.nonce = None
        self.key = None
        self.watermark = None
        self.latentlength = 4 * 64 * 64
        self.marklength = self.latentlength//(self.ch * self.hw * self.hw)

        self.threshold = 1 if self.hw == 1 and self.ch == 1 else self.ch * self.hw * self.hw // 2
        self.tp_onebit_count = 0
        self.tp_bits_count = 0
        self.tau_onebit = None
        self.tau_bits = None
        self.keepuser=keepuser

        for i in range(self.marklength):
            fpr_onebit = betainc(i+1, self.marklength-i, 0.5)
            fpr_bits = betainc(i+1, self.marklength-i, 0.5) * user_number
            if fpr_onebit <= fpr and self.tau_onebit is None:
                self.tau_onebit = i / self.marklength
            if fpr_bits <= fpr and self.tau_bits is None:
                self.tau_bits = i / self.marklength

        self.key = get_random_bytes(32)
        self.nonce = get_random_bytes(12)
        self.watermark = torch.randint(0, 2, [1, 4 // self.ch, 64 // self.hw, 64 // self.hw]).cuda()

    def stream_key_encrypt(self, sd):
        if not self.keepuser:
            self.key = get_random_bytes(32)
            self.nonce = get_random_bytes(12)
        cipher = ChaCha20.new(key=self.key, nonce=self.nonce)
        m_byte = cipher.encrypt(np.packbits(sd).tobytes())
        m_bit = np.unpackbits(np.frombuffer(m_byte, dtype=np.uint8))
        return m_bit

    def truncSampling(self, message):
        z = np.zeros(self.latentlength)
        denominator = 2.0
        ppf = [norm.ppf(j / denominator) for j in range(int(denominator) + 1)]
        for i in range(self.latentlength):
            dec_mes = reduce(lambda a, b: 2 * a + b, message[i : i + 1])
            dec_mes = int(dec_mes)
            z[i] = truncnorm.rvs(ppf[dec_mes], ppf[dec_mes + 1])
        z = torch.from_numpy(z).reshape(1, 4, 64, 64).half()
        return z.cuda()

    def create_watermark_and_return_w(self):
        if not self.keepuser:
            self.watermark = torch.randint(0, 2, [1, 4 // self.ch, 64 // self.hw, 64 // self.hw]).cuda()
        sd = self.watermark.repeat(1,self.ch,self.hw,self.hw)
        m = self.stream_key_encrypt(sd.flatten().cpu().numpy())
        w = self.truncSampling(m)
        return w

    def stream_key_decrypt(self, reversed_m):
        cipher = ChaCha20.new(key=self.key, nonce=self.nonce)
        sd_byte = cipher.decrypt(np.packbits(reversed_m).tobytes())
        sd_bit = np.unpackbits(np.frombuffer(sd_byte, dtype=np.uint8))
        sd_tensor = torch.from_numpy(sd_bit).reshape(1, 4, 64, 64).to(torch.uint8)
        return sd_tensor.cuda()

    def diffusion_inverse(self,watermark_r):
        ch_stride = 4 // self.ch
        hw_stride = 64 // self.hw
        ch_list = [ch_stride] * self.ch
        hw_list = [hw_stride] * self.hw
        split_dim1 = torch.cat(torch.split(watermark_r, tuple(ch_list), dim=1), dim=0)
        split_dim2 = torch.cat(torch.split(split_dim1, tuple(hw_list), dim=2), dim=0)
        split_dim3 = torch.cat(torch.split(split_dim2, tuple(hw_list), dim=3), dim=0)
        vote = torch.sum(split_dim3, dim=0).clone()
        vote[vote <= self.threshold] = 0
        vote[vote > self.threshold] = 1
        return vote

    def eval_watermark(self, reversed_w):
        reversed_m = (reversed_w > 0).int()
        reversed_sd = self.stream_key_decrypt(reversed_m.flatten().cpu().numpy())
        reversed_watermark = self.diffusion_inverse(reversed_sd)
        correct = (reversed_watermark == self.watermark).float().mean().item()
        if correct >= self.tau_onebit:
            self.tp_onebit_count = self.tp_onebit_count+1
        if correct >= self.tau_bits:
            self.tp_bits_count = self.tp_bits_count + 1
        print(correct)
        return correct

    def get_tpr(self):
        return self.tp_onebit_count, self.tp_bits_count





class TensorGenerator:
    def __init__(self, batch_size=1, channels=4, height=64, width=56):
        # Initialize the object with tensor sizes
        self.batch_size = batch_size
        self.channels = channels
        self.height = height  # Now height = 64
        self.width = width    # Now width = 56

        # Generate selected pairs for each channel
        self.selected_pairs_per_channel = self.generate_selected_pairs()

        # Reset with random B and binary tensor `key`
        self.reset(torch.randint(0, 2, (self.batch_size, self.channels, self.height, self.width)), 
                   torch.randint(0, 2, (self.batch_size, self.channels, 4, 1)))  # (1,4,4,1)

    def reset(self, B, key):
        """Resets the binary tensor and computes the corresponding sets."""
        self.B = B
        self.binary_tensor = key
        self.decimal_indices = self.convert_binary_to_index(self.binary_tensor)
        self.masked_cols_per_channel = self.get_masked_cols(self.selected_pairs_per_channel, self.decimal_indices)
        self.calculate_float_tensor()

    # ==============================================
    # 1. Generate random selected pairs (2 blocks out of 7 per channel)
    # ==============================================
    def generate_selected_pairs(self, num_blocks=7, num_samples=16):
        """Generate different (block1, block2) pairs for each channel."""
        block_combinations = list(combinations(range(num_blocks), 2))
        return [torch.tensor(random.sample(block_combinations, num_samples)) for _ in range(self.channels)]

    # ==============================================
    # 2. Convert binary tensor (1,4,4,1) to decimal indices (0-15)
    # ==============================================
    def convert_binary_to_index(self, binary_tensor):
        """Convert 4-bit binary to decimal indices (0-15)."""
        binary_tensor_flat = binary_tensor.squeeze(-1)  # (1,4,4)
        return (binary_tensor_flat * torch.tensor([8, 4, 2, 1])).sum(dim=-1)  # (1,4)

    # ==============================================
    # 3. Get masked column indices based on selected block pairs
    # ==============================================
    def get_masked_cols(self, selected_pairs_per_channel, decimal_indices):
        """Finds masked column indices based on block selection for each channel."""
        blocks = [list(range(i * 8, (i + 1) * 8)) for i in range(7)]  # 7 blocks of 8 columns each
        masked_cols_per_channel = {}

        for c in range(self.channels):  # Channel loop
            selected_blocks = selected_pairs_per_channel[c][decimal_indices[0, c]]  # (block1, block2)
            col_indices = [blocks[block] for block in selected_blocks]  # Convert blocks to column indices
            masked_cols_per_channel[c] = torch.tensor(col_indices).flatten().tolist()  # Flatten to 1D list

        return masked_cols_per_channel

    # ==============================================
    # 4. Generate the final float tensor
    # ==============================================
    def generate_float_tensor(self):
        """Generates (1,4,64,56) float tensor using B and masked column information."""
        output_tensor = torch.zeros(self.batch_size, self.channels, self.height, self.width)  # Initialize tensor

        # Convert sets to lists to ensure unique sampling
        positive_list = list(self.positive_set)
        negative_list = list(self.negative_set)
        remainder_list = list(self.remainder_set)

        for b in range(self.batch_size):  # Batch loop
            for c in range(self.channels):  # Channel loop
                for h in range(self.height):  # Row loop
                    for w in range(self.width):  # Column loop
                        if (h < self.height // 2 and w == self.masked_cols_per_channel[c][0]) or \
                           (h >= self.height // 2 and w == self.masked_cols_per_channel[c][1]):
                            value = remainder_list.pop()
                        else:
                            if self.B[b, c, h, w] == 1:  # B == 1 → positive set
                                value = positive_list.pop()
                            else:  # B == 0 → negative set
                                value = negative_list.pop()

                        output_tensor[b, c, h, w] = value  # Assign sampled value


        return output_tensor

    # ==============================================
    # 5. Generate Gaussian samples for the positive, negative, and remainder sets
    # ==============================================
    def generate_gaussian_samples(self, p, n):
        """Generate and split Gaussian samples into positive, negative, and remaining samples."""
        while True:
            # Generate 14336 random floats from a standard Gaussian distribution
            samples = np.random.randn(14336)

            # Separate positive and negative samples
            positive_samples = samples[samples > 0]
            negative_samples = samples[samples < 0]

            # Check if we have enough positive and negative samples
            if len(positive_samples) >= p and len(negative_samples) >= n:
                break

        # Get the max p positive and min n negative
        positive_part = np.partition(positive_samples, -p)[-p:]
        negative_part = np.partition(negative_samples, n)[:n]

        # The remaining samples
        remaining_samples = np.setdiff1d(samples, np.concatenate([positive_part, negative_part]))

        # Return the three parts
        return positive_part, negative_part, remaining_samples

    # ==============================================
    # 6. Calculate the positive, negative, and remainder sets based on the binary tensor
    # ==============================================
    def calculate_float_tensor(self):
        """Calculate the sets for positive, negative, and remainder samples."""
        positive_num = 0
        negative_num = 0

        for b in range(self.batch_size):  # Batch loop
            for c in range(self.channels):  # Channel loop
                for h in range(self.height):  # Row loop
                    for w in range(self.width):  # Column loop
                        if (h < self.height // 2 and w == self.masked_cols_per_channel[c][0]) or \
                           (h >= self.height // 2 and w == self.masked_cols_per_channel[c][1]):  # Masked columns → remainder set
                            continue
                        else:
                            if self.B[b, c, h, w] == 1:  # B == 1 → positive set
                                positive_num += 1
                            else:  # B == 0 → negative set
                                negative_num += 1

        # Generate the sets based on the counts
        self.positive_set, self.negative_set, self.remainder_set = self.generate_gaussian_samples(positive_num, negative_num)

    def mask_tensor(self, w,key):
        key = key.unsqueeze(0)
        decimal_indices = self.convert_binary_to_index(key)
        masked_cols_per_channel = self.get_masked_cols(self.selected_pairs_per_channel, decimal_indices)

        for b in range(self.batch_size):  # Batch loop
            for c in range(self.channels):  # Channel loop
                for h in range(self.height):  # Row loop
                    for w_idx in range(w.shape[-1]):  # Column loop
                        if (h < self.height // 2 and w_idx == masked_cols_per_channel[c][0]) or \
                           (h >= self.height // 2 and w_idx == masked_cols_per_channel[c][1]):  # Split masking
                            w[b, c, h, w_idx] = 0.0  # Mask the selected positions

        return w





class Dynamic_Gaussian_Shading_chacha:
    def __init__(self, ch_factor, hw_factor, fpr, user_number , keepuser=False):
        self.ch = ch_factor
        self.hw = hw_factor
        self.nonce = None
        self.key = None
        self.watermark = None
        self.latentlength = 4 * 64 * 64
        self.marklength = self.latentlength//(self.ch * self.hw * self.hw)

        self.threshold = 1 if self.hw == 1 and self.ch == 1 else self.ch * self.hw * self.hw // 2
        self.tp_onebit_count = 0
        self.tp_bits_count = 0
        self.tau_onebit = None
        self.tau_bits = None
        self.keepuser=keepuser

        for i in range(self.marklength):
            fpr_onebit = betainc(i+1, self.marklength-i, 0.5)
            fpr_bits = betainc(i+1, self.marklength-i, 0.5) * user_number
            if fpr_onebit <= fpr and self.tau_onebit is None:
                self.tau_onebit = i / self.marklength
            if fpr_bits <= fpr and self.tau_bits is None:
                self.tau_bits = i / self.marklength

        self.key = get_random_bytes(32)
        self.nonce = get_random_bytes(12)
        self.watermark = torch.randint(0, 2, [1, 4 // self.ch, 64 // self.hw, 64 // self.hw]).cuda()

        self.tg=TensorGenerator(batch_size=1, channels=4, height=64, width=56)

    def stream_key_encrypt(self, sd):
        cipher = ChaCha20.new(key=self.key, nonce=self.nonce)
        m_byte = cipher.encrypt(np.packbits(sd).tobytes())
        m_bit = np.unpackbits(np.frombuffer(m_byte, dtype=np.uint8))
        return m_bit

    def truncSampling(self, message):
        length=len(message)
        z = np.zeros(length)
        denominator = 2.0
        ppf = [norm.ppf(j / denominator) for j in range(int(denominator) + 1)]
        for i in range(length):
            dec_mes = reduce(lambda a, b: 2 * a + b, message[i : i + 1])
            dec_mes = int(dec_mes)
            z[i] = truncnorm.rvs(ppf[dec_mes], ppf[dec_mes + 1])
        z = torch.from_numpy(z).reshape(1, 4, 64, -1).half()
        return z.cuda()

    def create_watermark_and_return_w(self):
        if not self.keepuser:
            self.key = get_random_bytes(32)
            self.nonce = get_random_bytes(12)
        if not self.keepuser:
            self.watermark = torch.randint(0, 2, [1, 4 // self.ch, 64 // self.hw, 64 // self.hw]).cuda()
        
        random_mark= torch.randint(0, 2, [1, 4 // self.ch, 32 // self.hw, 1]).cuda()
        self.m=random_mark
        sd = random_mark.repeat(1,self.ch,self.hw*2,self.hw)
        random_m = self.stream_key_encrypt(sd.flatten().cpu().numpy())
        
        #merged_watermark=(self.watermark==random_mark.repeat(1,1,2,8)).int()
        sd = self.watermark.repeat(1,self.ch,self.hw,self.hw-1)
        m = self.stream_key_encrypt(sd.flatten().cpu().numpy())
               
        w1 = self.truncSampling(random_m)
        w2 = torch.from_numpy(m).reshape(1, 4, 64, -1).int()
        self.tg.reset(w2.cpu(),random_mark.cpu())
        w2=self.tg.generate_float_tensor().cuda().half()
        w0,w2=torch.split(w2,(24,32),dim=3)
        w=torch.cat((w0,w1,w2),dim=-1)
        
        
        self.w=(w>0).int().flatten()
        return w

    def stream_key_decrypt(self, reversed_m):
        cipher = ChaCha20.new(key=self.key, nonce=self.nonce)
        sd_byte = cipher.decrypt(np.packbits(reversed_m).tobytes())
        sd_bit = np.unpackbits(np.frombuffer(sd_byte, dtype=np.uint8))
        sd_tensor = torch.from_numpy(sd_bit).reshape(1, 4, 64, -1).to(torch.uint8)
        return sd_tensor.cuda()

    def diffusion_inverse(self,watermark_r):
        ch_stride = 4 // self.ch
        hw_stride = 64 // self.hw
        ch_list = [ch_stride] * self.ch
        h_list = [hw_stride] * self.hw
        w_list = [hw_stride] * (self.hw//2)
        split_dim1 = torch.cat(torch.split(watermark_r, tuple(ch_list), dim=1), dim=0)
        split_dim2 = torch.cat(torch.split(split_dim1, tuple(h_list), dim=2), dim=0)
        split_dim3 = torch.cat(torch.split(split_dim2, tuple(w_list), dim=3), dim=0)
        vote = torch.sum(split_dim3, dim=0).clone()
        vote[vote <= self.threshold//2] = 0
        vote[vote > self.threshold//2] = 1
        return vote
    
    def repeat_inverse(self,watermark_r,mode=False):
        ch_stride = 4 // self.ch
        hw_stride = 64 // self.hw
        ch_list = [ch_stride] * self.ch
        if mode:
            h_list = [hw_stride//2] * (self.hw*2) 
            w_list = [1] * (self.hw * 1)
        else:
            h_list = [hw_stride] * self.hw
            w_list = [hw_stride] * (self.hw-1)
        split_dim1 = torch.cat(torch.split(watermark_r, tuple(ch_list), dim=1), dim=0)
        split_dim2 = torch.cat(torch.split(split_dim1, tuple(h_list), dim=2), dim=0)
        split_dim3 = torch.cat(torch.split(split_dim2, tuple(w_list), dim=3), dim=0)
        vote = torch.sum(split_dim3, dim=0).clone()
        
        return (vote>0).int()

    def eval_watermark(self, reversed_w):
        w0,w1,w2=torch.split(reversed_w,(24,8,32),dim=3)
        w2=torch.cat((w0,w2),dim=3)
        reversed_m1 = (w1 > 0).int()
        weight_mask1=torch.abs(w1)
        reversed_sd1 = self.stream_key_decrypt(reversed_m1.flatten().cpu().numpy())
        #reversed_watermark = self.diffusion_inverse(reversed_sd)
        reversed_m1 = self.repeat_inverse((reversed_sd1.float()-0.5)*weight_mask1,True)
        rm=reversed_m1
        
        
        reversed_m2 = (w2 > 0).int()
        weight_mask2=torch.abs(w2)
        reversed_sd2 = self.stream_key_decrypt(reversed_m2.flatten().cpu().numpy())
        reversed_sd2 = ((reversed_sd2).float()-0.5)*weight_mask2
        reversed_sd2 = self.tg.mask_tensor(reversed_sd2,rm.cpu())
        reversed_m2 = self.repeat_inverse(reversed_sd2)
        
        reversed_watermark = reversed_m2
        
        
        correct = (reversed_watermark == self.watermark).float().mean().item()
        if correct >= self.tau_onebit:
            self.tp_onebit_count = self.tp_onebit_count+1
        if correct >= self.tau_bits:
            self.tp_bits_count = self.tp_bits_count + 1
            
        reversed_w = (reversed_w > 0).int().flatten()
        print(correct,torch.sum(reversed_w==self.w),torch.sum(self.m==rm)/len(rm.flatten()))
        return correct

    def get_tpr(self):
        return self.tp_onebit_count, self.tp_bits_count



class Gaussian_Shading:
    def __init__(self, ch_factor, hw_factor, fpr, user_number):
        self.ch = ch_factor
        self.hw = hw_factor
        self.key = None
        self.watermark = None
        self.latentlength = 4 * 64 * 64
        self.marklength = self.latentlength//(self.ch * self.hw * self.hw)

        self.threshold = 1 if self.hw == 1 and self.ch == 1 else self.ch * self.hw * self.hw // 2
        self.tp_onebit_count = 0
        self.tp_bits_count = 0
        self.tau_onebit = None
        self.tau_bits = None

        for i in range(self.marklength):
            fpr_onebit = betainc(i+1, self.marklength-i, 0.5)
            fpr_bits = betainc(i+1, self.marklength-i, 0.5) * user_number
            if fpr_onebit <= fpr and self.tau_onebit is None:
                self.tau_onebit = i / self.marklength
            if fpr_bits <= fpr and self.tau_bits is None:
                self.tau_bits = i / self.marklength

    def truncSampling(self, message):
        z = np.zeros(self.latentlength)
        denominator = 2.0
        ppf = [norm.ppf(j / denominator) for j in range(int(denominator) + 1)]
        for i in range(self.latentlength):
            dec_mes = reduce(lambda a, b: 2 * a + b, message[i : i + 1])
            dec_mes = int(dec_mes)
            z[i] = truncnorm.rvs(ppf[dec_mes], ppf[dec_mes + 1])
        z = torch.from_numpy(z).reshape(1, 4, 64, 64).half()
        return z.cuda()

    def create_watermark_and_return_w(self):
        self.key = torch.randint(0, 2, [1, 4, 64, 64]).cuda()
        self.watermark = torch.randint(0, 2, [1, 4 // self.ch, 64 // self.hw, 64 // self.hw]).cuda()
        sd = self.watermark.repeat(1,self.ch,self.hw,self.hw)
        m = ((sd + self.key) % 2).flatten().cpu().numpy()
        w = self.truncSampling(m)
        return w

    def diffusion_inverse(self,watermark_sd):
        ch_stride = 4 // self.ch
        hw_stride = 64 // self.hw
        ch_list = [ch_stride] * self.ch
        hw_list = [hw_stride] * self.hw
        split_dim1 = torch.cat(torch.split(watermark_sd, tuple(ch_list), dim=1), dim=0)
        split_dim2 = torch.cat(torch.split(split_dim1, tuple(hw_list), dim=2), dim=0)
        split_dim3 = torch.cat(torch.split(split_dim2, tuple(hw_list), dim=3), dim=0)
        vote = torch.sum(split_dim3, dim=0).clone()
        vote[vote <= self.threshold] = 0
        vote[vote > self.threshold] = 1
        return vote

    def eval_watermark(self, reversed_m):
        reversed_m = (reversed_m > 0).int()
        reversed_sd = (reversed_m + self.key) % 2
        reversed_watermark = self.diffusion_inverse(reversed_sd)
        correct = (reversed_watermark == self.watermark).float().mean().item()
        if correct >= self.tau_onebit:
            self.tp_onebit_count = self.tp_onebit_count+1
        if correct >= self.tau_bits:
            self.tp_bits_count = self.tp_bits_count + 1
        return correct

    def get_tpr(self):
        return self.tp_onebit_count, self.tp_bits_count
