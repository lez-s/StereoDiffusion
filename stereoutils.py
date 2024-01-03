import numpy as np
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from einops import rearrange,repeat
import gc
from torchvision.utils import save_image
import sys
sys.path.append('./stablediffusion')
from stablediffusion.ldm.models.diffusion.ddim import DDIMSampler

def stereo_shift_torch(input_images, depthmaps,sacle_factor=8,shift_both = False,stereo_offset_exponent=1.0):
    '''input: [B, C, H, W] depthmap: [B, H, W]'''

    def _norm_depth(depth,max_val=1):
        depth_min = depth.min()
        depth_max = depth.max()
        if depth_max - depth_min > np.finfo("float").eps:
            out = max_val * (depth - depth_min) / (depth_max - depth_min)
        else:
            out = np.zeros(depth.shape, dtype=depth.dtype)
        return out
    
    def _create_stereo(input_images,depthmaps,sacle_factor,stereo_offset_exponent):
        b, c, h, w = input_images.shape
        derived_image = torch.zeros_like(input_images)
        sacle_factor_px = (sacle_factor / 100.0) * input_images.shape[-1]
        # filled = torch.zeros(b * h * w, dtype=torch.uint8)
        if True:
            for batch in range(b):
                for row in range(h):
                    # Swipe order should ensure that pixels that are closer overwrite
                    # (at their destination) pixels that are less close
                    for col in range(w) if sacle_factor_px < 0 else range(w - 1, -1, -1):
                        col_d = col + int((depthmaps[batch,row,col] ** stereo_offset_exponent) * sacle_factor_px)
                        if 0 <= col_d < w:
                            derived_image[batch,:,row,col_d] = input_images[batch,:,row,col]
                            # filled[batch * h * w + row * w + col_d] = 1        
        return derived_image
    depthmaps = _norm_depth(depthmaps)
    
    if shift_both is False:
        left = input_images
        balance = 0
    else:
        balance = 0.5
        left = _create_stereo(input_images,depthmaps,+1 * sacle_factor * balance,stereo_offset_exponent)
    right = _create_stereo(input_images,depthmaps,-1 * sacle_factor * (1 - balance),stereo_offset_exponent)
    return torch.concat([left,right],axis=0)



class BNAttention():
    def __init__(self, start_step=4, total_steps=50, direction='uni'):

        self.total_steps = total_steps
        self.start_step = start_step
        self.cur_step = 0
        self.cur_att_layer = 0
        self.direction = direction

    
    def attn_batch(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        n_samples =attn.shape[0]//num_heads // 2
        q = rearrange(q, "(s b h) n d -> (b h) (s n) d", h=num_heads,b=n_samples)
        k = rearrange(k, "(s b h) n d -> (b h) (s n) d", h=num_heads,b=n_samples)
        v = rearrange(v, "(s b h) n d -> (b h) (s n) d", h=num_heads,b=n_samples)

        sim = torch.einsum("h i d, h j d -> h i j", q, k) * kwargs.get("scale")
        del q,k
        attn = sim.softmax(-1)
        out = torch.einsum("h i j, h j d -> h i d", attn, v)
        out = rearrange(out, "(b h) (s n) d -> (s b) n (h d)",b=n_samples,s=2,h=num_heads)
        return out
    
    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        if is_cross or (self.cur_step < self.start_step):
            out = torch.einsum('b i j, b j d -> b i d', attn, v)
            out = rearrange(out, '(b h) n d -> b n (h d)', h=num_heads)
            return out
        
        n_samples =attn.shape[0]//num_heads // 4
        qu, qc = q.chunk(2)
        ku, kc = k.chunk(2)
        vu, vc = v.chunk(2)
        attnu, attnc = attn.chunk(2)
        _num_heads = num_heads * n_samples
        if self.direction == 'bi':
            out_u = self.attn_batch(qu, ku, vu, sim, attnu, is_cross, place_in_unet, num_heads, **kwargs)
            out_c = self.attn_batch(qc, kc, vc, sim, attnc, is_cross, place_in_unet, num_heads, **kwargs)
        elif self.direction == 'uni':
            out_u = self.attn_batch(qu, ku[:_num_heads], vu[:_num_heads], sim[:_num_heads], attnu, is_cross, place_in_unet, num_heads, **kwargs)
            out_c = self.attn_batch(qc, kc[:_num_heads], vc[:_num_heads], sim[:_num_heads], attnc, is_cross, place_in_unet, num_heads, **kwargs)
        out = torch.cat([out_u, out_c], dim=0)
        return out

    def __call__(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        out = self.forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)
        self.cur_att_layer += 1
        self.cur_step = self.cur_att_layer//32
        return out


def regiter_attention_editor_diffusers(model, editor):
    """
    refer from [Prompt-to-Prompt]
    """
    def ca_forward(self, place_in_unet):
        def forward(x, encoder_hidden_states=None, attention_mask=None, context=None, mask=None):
            if encoder_hidden_states is not None:
                context = encoder_hidden_states
            if attention_mask is not None:
                mask = attention_mask

            to_out = self.to_out
            if isinstance(to_out, nn.modules.container.ModuleList):
                to_out = self.to_out[0]
            else:
                to_out = self.to_out

            h = self.heads
            q = self.to_q(x)
            is_cross = context is not None
            context = context if is_cross else x
            k = self.to_k(context)
            v = self.to_v(context)
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

            sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale

            if mask is not None:
                mask = rearrange(mask, 'b ... -> b (...)')
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = repeat(mask, 'b j -> (b h) () j', h=h)
                mask = mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~mask, max_neg_value)

            attn = sim.softmax(dim=-1)
            # the only difference
            out = editor(
                q, k, v, sim, attn, is_cross, place_in_unet,
                self.heads, scale=self.scale)

            return to_out(out)

        return forward

    def register_editor(net, count, place_in_unet):
        for name, subnet in net.named_children():
            # if net.__class__.__name__ == 'Attention':  # spatial Transformer layer
            if 'Attention' in net.__class__.__name__:
                net.forward = ca_forward(net, place_in_unet)
                return count + 1
            elif hasattr(net, 'children'):
                count = register_editor(subnet, count, place_in_unet)
        return count

    cross_att_count = 0
    try:
        sub_nets = model.model.diffusion_model.named_children()
    except: 
        sub_nets = model.unet.named_children()
    for net_name, net in sub_nets:
        if "down" in net_name or "input" in net_name:
            cross_att_count += register_editor(net, 0, "down")
        elif "mid" in net_name:
            cross_att_count += register_editor(net, 0, "mid")
        elif "up" in net_name or "output" in net_name:
            cross_att_count += register_editor(net, 0, "up")
    editor.num_att_layers = cross_att_count


def restore_attention(model):
    def ca_forward(self, place_in_unet):
        def forward(x, encoder_hidden_states=None, attention_mask=None, context=None, mask=None):

            if encoder_hidden_states is not None:
                context = encoder_hidden_states
            if attention_mask is not None:
                mask = attention_mask

            to_out = self.to_out
            if isinstance(to_out, nn.modules.container.ModuleList):
                to_out = self.to_out[0]
            else:
                to_out = self.to_out

            h = self.heads
            q = self.to_q(x)
            is_cross = context is not None
            context = context if is_cross else x
            k = self.to_k(context)
            v = self.to_v(context)
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

            sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale

            if mask is not None:
                mask = rearrange(mask, 'b ... -> b (...)')
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = repeat(mask, 'b j -> (b h) () j', h=h)
                mask = mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~mask, max_neg_value)

            attn = sim.softmax(dim=-1)
            # the only difference
            out = torch.einsum('b i j, b j d -> b i d', sim, v)
            out = rearrange(out, '(b h) n d -> b n (h d)', h=h)

            return to_out(out)

        return forward

    def register_editor(net, count, place_in_unet):
        for name, subnet in net.named_children():
            # if net.__class__.__name__ == 'Attention':  # spatial Transformer layer
            if 'Attention' in net.__class__.__name__:
                net.forward = ca_forward(net, place_in_unet)
                return count + 1
            elif hasattr(net, 'children'):
                count = register_editor(subnet, count, place_in_unet)
        return count

    cross_att_count = 0
    try:
        sub_nets = model.model.diffusion_model.named_children()
    except: 
        sub_nets = model.unet.named_children()
    for net_name, net in sub_nets:
        if "down" in net_name or "input" in net_name:
            cross_att_count += register_editor(net, 0, "down")
        elif "mid" in net_name:
            cross_att_count += register_editor(net, 0, "mid")
        elif "up" in net_name or "output" in net_name:
            cross_att_count += register_editor(net, 0, "up")


class StereoShiftSampler(DDIMSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ddimstep = 0
        self.disparity = None
        self.swapat = 10

    @torch.no_grad()
    def sample(self, *args, **kwargs):
        try:
            self.disparity = kwargs['disparity']
            self.shift_both = kwargs['shift_both']
            self.swapat = kwargs['swapat']
            self.deblur = kwargs['deblur']
            print(self.swapat)
        except:
            pass
        
        return super().sample(*args, **kwargs)

    @torch.no_grad()
    def p_sample_ddim(self, *args, **kwargs):
        x_prev, pred_x0 = super().p_sample_ddim(*args, **kwargs)
        if self.disparity is not None:
            if self.ddimstep==self.swapat:
                x = args[0]
                N = x.shape[0]//2
                self.disparity = torch.nn.functional.interpolate(self.disparity.unsqueeze(1),size=[64,64],mode="bicubic",align_corners=False,).squeeze(1)
                x_prev = stereo_shift_torch(x_prev[:N],self.disparity,shift_both=self.shift_both)
                pred_x0 = stereo_shift_torch(pred_x0[:N],self.disparity,shift_both=self.shift_both)
                # mask = x_prev[N:,0,...] != 0
                # mask = x_prev[N:,...] != [0,0,0,0]
                mask = torch.all(x_prev[N:,...] != 0, dim=1)

                self.mask = rearrange(mask,'b h w ->b () h w').repeat(1,4,1,1)
                if self.deblur: # aviod blurry
                    noise = torch.randn_like(x_prev)
                    x_prev[N:][~self.mask] = noise[N:][~self.mask]
                    pred_x0[N:][~self.mask] = noise[N:][~self.mask]

            if self.ddimstep > self.swapat and self.ddimstep % 10 ==0:
                N = x_prev.shape[0]//2
                x_prev_new = stereo_shift_torch(x_prev[:N],self.disparity,shift_both=self.shift_both)
                x_prev[N:][self.mask] = x_prev_new[N:][self.mask]
                # pred_x0_new = stereo_shift_torch(pred_x0[:N],self.disparity,shift_both=self.shift_both) 
                # pred_x0[N:][self.mask] = pred_x0_new[N:][self.mask]
            self.ddimstep += 1
        torch.cuda.empty_cache()
        gc.collect()

        return x_prev, pred_x0

    @torch.no_grad()
    def decode(self,*args,**kwargs):
        try:
            self.disparity = kwargs['disparity']
            self.shift_both = kwargs['shift_both']
            self.swapat = kwargs['swapat']
            self.deblur = kwargs['deblur']
            del kwargs['disparity'],kwargs['shift_both'],kwargs['swapat'],kwargs['deblur']
        except:
            pass
        return super().decode(*args, **kwargs)
    
    
def rearrange_img(img):
    n = img.shape[0]
    img_li = []
    for i in range(n//2):
        tmp = torch.concat((img[i],img[i+n//2]),axis=2)
        img_li.append(tmp)
    new_img = torch.stack(img_li,0)
    return new_img