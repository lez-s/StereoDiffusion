
import argparse, os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange, repeat
import sys
from typing import Optional, Union, Tuple, List
sys.path.append('./stablediffusion')
sys.path.append('./DPT')
from stablediffusion.ldm.util import instantiate_from_config
from DPT.dpt.models import DPTDepthModel
from stereoutils import *
sys.path.append('./prompt-to-prompt')
import ptp_utils 
from skimage.transform import resize
# import p2putil
from diffusers import StableDiffusionPipeline, DDIMScheduler
# torch.set_grad_enabled(False)
import torch.nn.functional as nnf
import abc
import seq_aligner
import shutil
from torch.optim.adam import Adam
import torchvision

scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
GUIDANCE_SCALE = 7.5
NUM_DDIM_STEPS = 50
MY_TOKEN = ''
ldm_stable = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=MY_TOKEN, scheduler=scheduler).to(device)
try:
    ldm_stable.disable_xformers_memory_efficient_attention()
except AttributeError:
    print("Attribute disable_xformers_memory_efficient_attention() is missing")
tokenizer = ldm_stable.tokenizer


class EmptyControl:
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        return attn

class NullInversion:
    
    def prev_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        return prev_sample
    
    def next_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        timestep, next_timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample
    
    def get_noise_pred_single(self, latents, t, context):
        noise_pred = self.model.unet(latents, t, encoder_hidden_states=context)["sample"]
        return noise_pred

    def get_noise_pred(self, latents, t, is_forward=True, context=None):
        latents_input = torch.cat([latents] * 2)
        if context is None:
            context = self.context
        guidance_scale = 1 if is_forward else GUIDANCE_SCALE
        noise_pred = self.model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
        if is_forward:
            latents = self.next_step(noise_pred, t, latents)
        else:
            latents = self.prev_step(noise_pred, t, latents)
        return latents

    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        latents = 1 / 0.18215 * latents.detach()
        image = self.model.vae.decode(latents)['sample']
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        return image

    @torch.no_grad()
    def image2latent(self, image):
        with torch.no_grad():
            if type(image) is Image:
                image = np.array(image)
            if type(image) is torch.Tensor and image.dim() == 4:
                latents = image
            else:
                image = torch.from_numpy(image).float() / 127.5 - 1
                image = image.permute(2, 0, 1).unsqueeze(0).to(device)
                latents = self.model.vae.encode(image)['latent_dist'].mean
                latents = latents * 0.18215
        return latents

    @torch.no_grad()
    def init_prompt(self, prompt: str):
        uncond_input = self.model.tokenizer(
            [""], padding="max_length", max_length=self.model.tokenizer.model_max_length,
            return_tensors="pt"
        )
        uncond_embeddings = self.model.text_encoder(uncond_input.input_ids.to(self.model.device))[0]
        text_input = self.model.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.model.device))[0]
        self.context = torch.cat([uncond_embeddings, text_embeddings])
        self.prompt = prompt

    @torch.no_grad()
    def ddim_loop(self, latent):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        all_latent = [latent]
        latent = latent.clone().detach()
        for i in range(NUM_DDIM_STEPS):
            t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
            noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings)
            latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)
        return all_latent

    @property
    def scheduler(self):
        return self.model.scheduler

    @torch.no_grad()
    def ddim_inversion(self, image):
        latent = self.image2latent(image)
        image_rec = self.latent2image(latent)
        ddim_latents = self.ddim_loop(latent)
        return image_rec, ddim_latents

    def null_optimization(self, latents, num_inner_steps, epsilon):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        uncond_embeddings_list = []
        latent_cur = latents[-1]
        # bar = tqdm(total=num_inner_steps * NUM_DDIM_STEPS)
        for i in tqdm(range(NUM_DDIM_STEPS)):
            uncond_embeddings = uncond_embeddings.clone().detach()
            uncond_embeddings.requires_grad = True
            optimizer = Adam([uncond_embeddings], lr=1e-2 * (1. - i / 100.))
            latent_prev = latents[len(latents) - i - 2]
            t = self.model.scheduler.timesteps[i]
            with torch.no_grad():
                noise_pred_cond = self.get_noise_pred_single(latent_cur, t, cond_embeddings)
            for j in range(num_inner_steps):
                noise_pred_uncond = self.get_noise_pred_single(latent_cur, t, uncond_embeddings)
                noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (noise_pred_cond - noise_pred_uncond)
                latents_prev_rec = self.prev_step(noise_pred, t, latent_cur)
                loss = nnf.mse_loss(latents_prev_rec, latent_prev)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_item = loss.item()
                # bar.update()
                if loss_item < epsilon + i * 2e-5:
                    break
            # for j in range(j + 1, num_inner_steps):
                # bar.update()
            uncond_embeddings_list.append(uncond_embeddings[:1].detach())
            with torch.no_grad():
                context = torch.cat([uncond_embeddings, cond_embeddings])
                latent_cur = self.get_noise_pred(latent_cur, t, False, context)
        # bar.close()
        return uncond_embeddings_list
    
    def invert(self, image, prompt: str, offsets=(0,0,0,0), num_inner_steps=10, early_stop_epsilon=1e-5, verbose=False):
        self.init_prompt(prompt)
        ptp_utils.register_attention_control(self.model, None)
        if isinstance(image, str):
            image_gt = load_512(image, *offsets)
        elif isinstance(image, np.ndarray):
            image_gt = resize(image, (512, 512))
            if image_gt.max()<=1:
                image_gt = (image_gt * 255).astype(np.uint8)
        else:
            raise ValueError("image_path must be either a path to an image or a numpy array")
        if verbose:
            print("DDIM inversion...")
        image_rec, ddim_latents = self.ddim_inversion(image_gt)
        if verbose:
            print("Null-text optimization...")
        uncond_embeddings = self.null_optimization(ddim_latents, num_inner_steps, early_stop_epsilon)
        return (image_gt, image_rec), ddim_latents[-1], uncond_embeddings
        
    
    def __init__(self, model):
        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False,
                                  set_alpha_to_one=False)
        self.model = model
        self.tokenizer = self.model.tokenizer
        self.model.scheduler.set_timesteps(NUM_DDIM_STEPS)
        self.prompt = None
        self.context = None




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str, required=True,help="path to image")
    parser.add_argument("--depthmodel_path",type=str,required=True,help='path of depth model')
    parser.add_argument(
        "--deblur",
        action='store_true',
        default=False,
    )
    parser.add_argument(
        "--scale_factor",
        type=float,
        default=9.0,
        help="scale factor of disparity",
    )
    parser.add_argument(
        "--direction",
        type=str,
        choices=["uni", "bi"],
        default="uni"
    )
    return parser.parse_args()

def load_512(image_path, left=0, right=0, top=0, bottom=0):
    if type(image_path) is str:
        image = np.array(Image.open(image_path))[:, :, :3]
    else:
        image = image_path
    h, w, c = image.shape
    if w != h:
        left = min(left, w-1)
        right = min(right, w - left - 1)
        top = min(top, h - left - 1)
        bottom = min(bottom, h - top - 1)
        image = image[top:h-bottom, left:w-right]
        h, w, c = image.shape
        if h < w:
            offset = (w - h) // 2
            image = image[:, offset:offset + h]
        elif w < h:
            offset = (h - w) // 2
            image = image[offset:offset + w]
    image = np.array(Image.fromarray(image).resize((512, 512)))
    return image

def _norm_depth(depth,max_val=1):
    depth_min = depth.min()
    depth_max = depth.max()
    if depth_max - depth_min > torch.finfo(torch.float32).eps:
        out = max_val * (depth - depth_min) / (depth_max - depth_min)
    else:
        out = torch.zeros(depth.shape, dtype=depth.dtype)
    return out

def run_inv_sd(image,args):
    depthmodel_path = args.depthmodel_path
    deblur = args.deblur

    device = torch.device("cuda")
    null_inversion = NullInversion(ldm_stable)
    prompt = ""
    prompts = [prompt]
    (image_gt, image_enc), x_t, uncond_embeddings = null_inversion.invert(image, prompt, offsets=(0,0,200,0), verbose=True)
    # del null_inversion
    
    # controller = AttentionStore()
    # image_inv, x_t = run_and_display(prompts, controller, run_baseline=False, latent=x_t, uncond_embeddings=uncond_embeddings, verbose=False)

    print("showing from left to right: the ground truth image, the vq-autoencoder reconstruction, the null-text inverted image")
    # ptp_utils.view_images([image_gt, image_enc, image_inv[0]])
    # show_cross_attention(controller, 16, ["up", "down"])
        
    net_w = net_h = 384
    depthmodel = DPTDepthModel(
        path=depthmodel_path,
        backbone="vitb_rn50_384",
        non_negative=True,
        enable_attention_hooks=False,
    ).cuda()
    image_gt_ = torch.tensor(np.expand_dims(image_gt/255,0).transpose(0,3,1,2)/255,device=device,dtype=torch.float32)
    with torch.no_grad():
        prediction = depthmodel.forward(image_gt_)
    disp = _norm_depth(prediction)
    del depthmodel

    
    @torch.no_grad()
    def text2stereoimage_ldm_stable(
        model,
        prompt:  List[str],
        controller,
        num_inference_steps: int = 50,
        guidance_scale: Optional[float] = 7.5,
        generator: Optional[torch.Generator] = None,
        latent: Optional[torch.FloatTensor] = None,
        uncond_embeddings=None,
        start_time=50,
        return_type='image',disparity=disp):


        sa = 10
        editor = BNAttention(start_step=sa,direction=args.direction)
        regiter_attention_editor_diffusers(model, editor)

        batch_size = len(prompt)
        # ptp_utils.register_attention_control(model, controller)
        height = width = 512
        # controller = editor
        
        text_input = model.tokenizer(
            prompt,
            padding="max_length",
            max_length=model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
        max_length = text_input.input_ids.shape[-1]
        if uncond_embeddings is None:
            uncond_input = model.tokenizer(
                [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
            )
            uncond_embeddings_ = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
        else:
            uncond_embeddings_ = None

        latent, latents = ptp_utils.init_latent(latent, model, height, width, generator, batch_size)
        model.scheduler.set_timesteps(num_inference_steps)
        for i, t in enumerate(tqdm(model.scheduler.timesteps[-start_time:])):
            if uncond_embeddings_ is None:
                context = torch.cat([uncond_embeddings[i].expand(*text_embeddings.shape), text_embeddings])
            else:
                context = torch.cat([uncond_embeddings_, text_embeddings])
            latents = ptp_utils.diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource=False)
            if i == 10:
                if isinstance(disparity,torch.Tensor):
                    disparity = torch.nn.functional.interpolate(disparity.unsqueeze(1),size=[64,64],mode="bicubic",align_corners=False,).squeeze(1)
                elif isinstance(disparity,np.ndarray):
                    disparity = resize(disparity,(64,64))
                # latents = stereo_shift_torch(latents[:1],disparity,stereo_balance=-1)
                sacle_factor = 8
                latents_ts = stereo_shift_torch(latents[:1], disparity, sacle_factor=sacle_factor)[1:]
                # latents_np_ = process_pixels_rgba_naive(latents_np[0])
                # latents_ts =torch.tensor(rearrange(latents_np,'b h w c -> b c h w'),device=device)
                latents = torch.cat([latents[:1],latents_ts],0)
                mask = latents_ts[:,0,...] != 0
                mask = rearrange(mask,'b h w ->b () h w').repeat(1,4,1,1)
                nosie = torch.randn_like(latents)
                
                if deblur: # aviod blurry
                    latents[1:][~mask] = nosie[1:][~mask]
                    latents[1:][mask] = latents_ts[mask]

            if  (i > 10 and i % 10 == 0):
                latents_ts = stereo_shift_torch(latents[:1], disparity, sacle_factor=sacle_factor)[1:]
                # latents_ts =torch.tensor(rearrange(latents_r_np,'b h w c -> b c h w'),device=device)
                latents[1:][mask] = latents_ts[mask]
                # latents[1:][mask] = replacement
                # import pdb;pdb.set_trace()
                
            
        if return_type == 'image':
            image = ptp_utils.latent2image(model.vae, latents)
        else:
            image = latents
        return image, latent
    # image, latent = text2stereoimage_ldm_stable(ldm_stable, prompts*2, controller,uncond_embeddings = uncond_embeddings,latent=torch.concat([x_t,x_t],0))
    # image_ = rearrange(image,'b h w c->h (b w) c')

    controller = EmptyControl()
    image, latent = text2stereoimage_ldm_stable(ldm_stable, prompts*2, controller,uncond_embeddings = uncond_embeddings,latent=torch.concat([x_t,x_t],0),disparity=disp)
    image_pair = rearrange(image,'b h w c->h (b w) c')


    return image, image_pair

if __name__ == "__main__":

    args = parse_args()
    image  = load_512(args.img_path)
    out_image,image_pair = run_inv_sd(image,args)
    Image.fromarray(image_pair).save(os.path.join('outputs',f'{args.img_path.split("/")[-1]}'))
