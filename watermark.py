import torch
from scipy.stats import norm,truncnorm
from functools import reduce
from scipy.special import betainc
import numpy as np
from Crypto.Cipher import ChaCha20
from Crypto.Random import get_random_bytes
from tqdm import  tqdm

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
        
        merged_watermark=(self.watermark==random_mark.repeat(1,1,2,8)).int()
        sd = merged_watermark.repeat(1,self.ch,self.hw,self.hw-1)
        m = self.stream_key_encrypt(sd.flatten().cpu().numpy())
        
        
        
        w1 = self.truncSampling(random_m)
        w2 = self.truncSampling(m)
        w0,w2=torch.split(w2,(24,32),dim=3)
        w=torch.cat((w0,w1,w2),dim=-1)
        
        torch.save(w,'vit.pt')
        
        
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
        reversed_m2 = self.repeat_inverse(((reversed_sd2).float()-0.5)*weight_mask2)
        
        reversed_watermark = (reversed_m1.repeat(1,1,2,8)==reversed_m2).int()
        
        
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

class TreeRing:
    def __init__(self, fpr, user_number):
        self.key = None
        self.watermark = None
        self.latentlength = 4 * 64 * 64



        self.tp_onebit_count = 0
        self.tp_bits_count = 0
        self.tau_onebit = None
        self.tau_bits = None
        self.device='cuda'

        self.gt = self.get_watermarking_pattern()
        self.mask=self.get_watermarking_mask(self.gt)
        self.marklength = torch.sum(self.mask).item()

        for i in range(self.marklength):
                fpr_onebit = betainc(i + 1, self.marklength - i, 0.5)
                fpr_bits = betainc(i + 1, self.marklength - i, 0.5) * user_number
                if fpr_onebit <= fpr and self.tau_onebit is None:
                    self.tau_onebit = i / self.marklength
                if fpr_bits <= fpr and self.tau_bits is None:
                    self.tau_bits = i / self.marklength

    def circle_mask(self,size=64, r=10, x_offset=0, y_offset=0):
        # reference: https://stackoverflow.com/questions/69687798/generating-a-soft-circluar-mask-using-numpy-python-3
        x0 = y0 = size // 2
        x0 += x_offset
        y0 += y_offset
        y, x = np.ogrid[:size, :size]
        y = y[::-1]

        return ((x - x0) ** 2 + (y - y0) ** 2) <= r ** 2

    def get_watermarking_mask(self,init_latents_w):
        watermarking_mask = torch.zeros(init_latents_w.shape, dtype=torch.bool).to(self.device)
        np_mask = self.circle_mask(init_latents_w.shape[-1], r=10)
        torch_mask = torch.tensor(np_mask).to(self.device)

        watermarking_mask[:, 3] = torch_mask


        return watermarking_mask

    def get_watermarking_pattern(self, shape=[1,4,64,64]):

        gt_init = torch.randn(*shape, device=self.device).half()
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))
        gt_patch[:] = gt_patch[0]

        return gt_patch

    def inject_watermark(self,init_latents_w, watermarking_mask, gt_patch):
        init_latents_w_fft = torch.fft.fftshift(torch.fft.fft2(init_latents_w), dim=(-1, -2))

        init_latents_w_fft[watermarking_mask] = gt_patch[watermarking_mask].clone()

        init_latents_w = torch.fft.ifft2(torch.fft.ifftshift(init_latents_w_fft, dim=(-1, -2))).real

        return init_latents_w

    def create_watermark_and_return_w(self , shape=[1,4,64,64]):
        init_latents_w = torch.randn(*shape, device=self.device).half()
        return  self.inject_watermark(init_latents_w,self.mask,self.gt)



    def eval_watermark(self, reversed_latents_w):
        reversed_latents_w_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents_w), dim=(-1, -2))
        target_patch = self.gt
        correct = torch.abs(reversed_latents_w_fft[self.mask] - target_patch[self.mask]).mean().item()
        if correct >= self.tau_onebit:
            self.tp_onebit_count = self.tp_onebit_count + 1
        if correct >= self.tau_bits:
            self.tp_bits_count = self.tp_bits_count + 1
        return correct

    def get_tpr(self):
        return self.tp_onebit_count, self.tp_bits_count
import cv2
from imwatermark import WatermarkEncoder
from imwatermark import WatermarkDecoder
class Image_Watermark:
    def __init__(self,  mode,fpr, user_number):
        self.key = None
        self.watermark = None
        self.latentlength = 4 * 64 * 64
        self.marklength = 256
        self.mode=mode


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


    def create_watermark(self,image_w):
        self.watermark=torch.randint(0, 2, [1, 4, 8, 8]).flatten()
        if self.mode==2:
            self.watermark = torch.randint(0, 2, [1, 4, 8]).flatten()
        if self.mode==0:
            wm=np.packbits(self.watermark.numpy()).tobytes()
            encoder = WatermarkEncoder()
            encoder.set_watermark('bytes', wm)
            image_w = encoder.encode(image_w, 'dwtDct')
        elif self.mode==1:
            wm=np.packbits(self.watermark.numpy()).tobytes()
            encoder = WatermarkEncoder()
            encoder.set_watermark('bytes', wm)
            image_w = encoder.encode(image_w, 'dwtDctSvd')
        elif self.mode==2:
            wm=np.packbits(self.watermark.numpy()).tobytes()
            encoder = WatermarkEncoder()
            encoder.loadModel()
            encoder.set_watermark('bytes', wm)
            image_w = encoder.encode(image_w, 'rivaGan')
        return image_w


    def eval_watermark(self, image_w):
        if self.mode==0:
            decoder = WatermarkDecoder('bytes', 256)
            wm = decoder.decode(image_w, 'dwtDct')
            wm = np.frombuffer(wm, dtype=np.uint8)
            watermark = torch.tensor(np.unpackbits(wm))
        elif self.mode==1:
            decoder = WatermarkDecoder('bytes', 256)
            wm = decoder.decode(image_w, 'dwtDctSvd')
            wm = np.frombuffer(wm, dtype=np.uint8)
            watermark = torch.tensor(np.unpackbits(wm))
        elif self.mode==2:
            decoder = WatermarkDecoder('bytes', 32)
            decoder.loadModel()
            wm = decoder.decode(image_w, 'rivaGan')
            wm = np.frombuffer(wm, dtype=np.uint8)
            watermark = torch.tensor(np.unpackbits(wm))

        correct = (watermark == self.watermark).float().mean().item()
        if correct >= self.tau_onebit:
            self.tp_onebit_count = self.tp_onebit_count+1
        if correct >= self.tau_bits:
            self.tp_bits_count = self.tp_bits_count + 1
        print(correct)
        return correct

    def get_tpr(self):
        return self.tp_onebit_count, self.tp_bits_count