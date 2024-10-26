import argparse
import copy
from tqdm import tqdm
import torch
from transformers import CLIPModel, CLIPTokenizer
from inverse_stable_diffusion import InversableStableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler, DDIMScheduler
import open_clip
from optim_utils import *
from io_utils import *
from image_utils import *
from watermark import *
import cv2
from imwatermark import WatermarkEncoder
from imwatermark import WatermarkDecoder
import numpy as np


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    scheduler = DPMSolverMultistepScheduler.from_pretrained(args.model_path, subfolder='scheduler')
    pipe = InversableStableDiffusionPipeline.from_pretrained(
            args.model_path,
            scheduler=scheduler,
            torch_dtype=torch.float16,
            revision='fp16',
    )
    pipe.safety_checker = None
    pipe = pipe.to(device)

    #reference model for CLIP Score
    if args.reference_model is not None:
        ref_model, _, ref_clip_preprocess = open_clip.create_model_and_transforms(args.reference_model,
                                                                                  pretrained=args.reference_model_pretrain,
                                                                                  device=device)
        ref_tokenizer = open_clip.get_tokenizer(args.reference_model)

    # dataset
    dataset, prompt_key = get_dataset(args)

    # class for watermark
    if args.chacha:
        watermark = Gaussian_Shading_Plus_chacha(args.channel_copy, args.hw_copy, args.fpr, args.user_number,True)
    else:
        #a simple implement,
        watermark = Gaussian_Shading(args.channel_copy, args.hw_copy, args.fpr, args.user_number)

    os.makedirs(args.output_path, exist_ok=True)

    # assume at the detection time, the original prompt is unknown
    tester_prompt = ''
    text_embeddings = pipe.get_text_embedding(tester_prompt)

    #acc
    acc = []
    #CLIP Features
    feas = []
    watermark = torch.randint(0, 2, [1, 4 ,8,8]).flatten()
    watermark32 = torch.randint(0, 2, [1, 4 ,8]).flatten()
    #test
    for i in tqdm(range(args.num)):
        seed = i + args.gen_seed
        current_prompt = dataset[args.fixed_idx][prompt_key]

        #generate with watermark
        set_random_seed(seed)
        #init_latents_w = watermark.create_watermark_and_return_w()
        init_latents_w = torch.randn((1,4,64,64)).cuda().half()
        outputs = pipe(
            current_prompt,
            num_images_per_prompt=1,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            height=args.image_length,
            width=args.image_length,
            latents=init_latents_w,
        )
        image_w = outputs.images[0]
        image_w=cv2.cvtColor(np.array(image_w), cv2.COLOR_RGB2BGR)
        if args.test_mode==0:
            wm=np.packbits(watermark.numpy()).tobytes()
            encoder = WatermarkEncoder()
            encoder.set_watermark('bytes', wm)
            image_w = encoder.encode(image_w, 'dwtDct')
        elif args.test_mode==1:
            wm=np.packbits(watermark.numpy()).tobytes()
            encoder = WatermarkEncoder()
            encoder.set_watermark('bytes', wm)
            image_w = encoder.encode(image_w, 'dwtDctSvd')
        elif args.test_mode==2:
            wm=np.packbits(watermark32.numpy()).tobytes()
            encoder = WatermarkEncoder()
            encoder.loadModel()
            encoder.set_watermark('bytes', wm)
            image_w = encoder.encode(image_w, 'rivaGan')
        
        image_w = Image.fromarray(cv2.cvtColor(image_w,cv2.COLOR_BGR2RGB))   

        
        #CLIP Score
        if args.reference_model is not None:
            fea = measure_diveristy([image_w], current_prompt, ref_model,
                                              ref_clip_preprocess,
                                              ref_tokenizer, device)
        else:
            fea = 0
        feas.append(fea)
    np.save('baseline'+str(args.test_mode)+'.npy', np.stack(feas))

    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gaussian Shading')
    parser.add_argument('--num', default=1000, type=int)
    parser.add_argument('--image_length', default=512, type=int)
    parser.add_argument('--guidance_scale', default=7.5, type=float)
    parser.add_argument('--num_inference_steps', default=50, type=int)
    parser.add_argument('--num_inversion_steps', default=None, type=int)
    parser.add_argument('--gen_seed', default=0, type=int)
    parser.add_argument('--channel_copy', default=1, type=int)
    parser.add_argument('--hw_copy', default=8, type=int)
    parser.add_argument('--user_number', default=1000000, type=int)
    parser.add_argument('--fpr', default=0.000001, type=float)
    parser.add_argument('--output_path', default='./output/')
    parser.add_argument('--chacha', action='store_false', help='chacha20 for cipher')
    parser.add_argument('--reference_model', default='ViT-g-14')
    parser.add_argument('--reference_model_pretrain', default='laion2b_s12b_b42k')
    parser.add_argument('--dataset_path', default='Gustavosta/Stable-Diffusion-Prompts')
    parser.add_argument('--model_path', default='stabilityai/stable-diffusion-2-1-base')


    
    parser.add_argument('--filename', default='origin', type=str)
    parser.add_argument('--fixed_idx', default=0, type=int)
    parser.add_argument('--test_mode', default=0, type=int)


    args = parser.parse_args()

    if args.num_inversion_steps is None:
        args.num_inversion_steps = args.num_inference_steps

    main(args)
