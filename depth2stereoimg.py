"""make variations of input image"""

import argparse, os
import PIL
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from pytorch_lightning import seed_everything
import torchvision.utils as vutils
import sys

sys.path.append('./stablediffusion')
sys.path.append('./DPT')
from stablediffusion.ldm.util import instantiate_from_config
from DPT.dpt.models import DPTDepthModel
from stablediffusion.ldm.data.util import AddMiDaS
from stereoutils import *
torch.set_grad_enabled(False)

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def load_img(path):
    image = Image.open(path).convert("RGB")
    w, h = image.size

    print(f"loaded input image of size ({w}, {h}) from {path}")
    image = image.resize((512, 512))
    return image

def make_batch_sd(
        image,
        txt,
        device,
        num_samples=1,
        model_type="dpt_hybrid"
):
    image = np.array(image.convert("RGB"))
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0
    # sample['jpg'] is tensor hwc in [-1, 1] at this point
    midas_trafo = AddMiDaS(model_type=model_type)
    batch = {
        "jpg": image,
        "txt": num_samples * [txt],
    }
    batch = midas_trafo(batch)
    batch["jpg"] = rearrange(batch["jpg"], 'h w c -> 1 c h w')
    batch["jpg"] = repeat(batch["jpg"].to(device=device),
                          "1 ... -> n ...", n=num_samples)
    batch["midas_in"] = repeat(torch.from_numpy(batch["midas_in"][None, ...]).to(
        device=device), "1 ... -> n ...", n=num_samples)
    return batch

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )

    parser.add_argument(
        "--init_img",
        type=str,
        nargs="?",
        default="ori_img/ori-2.png",
        help="path to the input image"
    )

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/depth2stereoimg-samples"
    )

    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )

    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across all samples ",
    )

    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )

    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor, most often 8 or 16",
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        default=2,
        help="how many samples to produce for each given prompt. A.k.a batch size",
    )

    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=9.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )

    parser.add_argument(
        "--strength",
        type=float,
        default=1,
        help="strength for noising/unnoising. 1.0 corresponds to full destruction of information in init image",
    )

    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="v2-midas-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--direction",
        type=str,
        choices=["uni", "bi"],
        default="uni"
    )
    parser.add_argument(
        "--deblur",
        action='store_true',
        default=False,
    )
    parser.add_argument(
        "--shift_both",
        action='store_true',
        default=False,
    )
    # parser.add_argument(
    #     "--no_full_sample",
    #     action='store_true',
    #     default=False,
    # )
    parser.add_argument("--depthmodel_path",type=str,required=True,help='path of depth model')
    return parser.parse_args()

def main(opt):
    # do_full_sample = False if opt.no_full_sample else True
    
    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    sampler = StereoShiftSampler(model, device=device)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)

    assert os.path.isfile(opt.init_img)
    init_image = load_img(opt.init_img) #.to(device)

    # sampler.make_schedule(ddim_num_steps=opt.ddim_steps, ddim_eta=opt.ddim_eta, verbose=False)

    assert 0. <= opt.strength <= 1., 'can only work with strength in [0.0, 1.0]'
    t_enc = int(opt.strength * opt.ddim_steps)
    print(f"target t_enc is {t_enc} steps")

    sampler.make_schedule(ddim_num_steps=opt.ddim_steps, ddim_eta=opt.ddim_eta, verbose=False)
    with torch.no_grad():
        with model.ema_scope():
            batch = make_batch_sd(init_image, txt=opt.prompt, device=device, num_samples=opt.n_samples)
            z = model.get_first_stage_encoding(model.encode_first_stage(batch[model.first_stage_key]))
            c = model.cond_stage_model.encode(batch["txt"])
            c_cat = list()
            for ck in model.concat_keys:
                cc = batch[ck]
                cc = model.depth_model(cc)
                depth_min, depth_max = torch.amin(cc, dim=[1, 2, 3], keepdim=True), torch.amax(cc, dim=[1, 2, 3],
                                                                                            keepdim=True)
                display_depth = (cc - depth_min) / (depth_max - depth_min)
                depth_image = Image.fromarray(
                    (display_depth[0, 0, ...].cpu().numpy() * 255.).astype(np.uint8))
                cc = torch.nn.functional.interpolate(
                    cc,
                    size=z.shape[2:],
                    mode="bicubic",
                    align_corners=False,
                )
                depth_min, depth_max = torch.amin(cc, dim=[1, 2, 3], keepdim=True), torch.amax(cc, dim=[1, 2, 3],
                                                                                            keepdim=True)
                cc = 2. * (cc - depth_min) / (depth_max - depth_min) - 1.
                c_cat.append(cc)
            c_cat = torch.cat(c_cat, dim=1)
            c_cat = torch.cat([c_cat, c_cat], dim=0)
            c = torch.cat([c, c], dim=0)
            # cond
            cond = {"c_concat": [c_cat], "c_crossattn": [c]}
            
            sa = 10
            # prediction = depthmodel.forward(init_image)
            editor = BNAttention(start_step=sa,direction=opt.direction)
            regiter_attention_editor_diffusers(model, editor)   

            uc_cross = model.get_unconditional_conditioning(opt.n_samples, "")
            uc_cross = torch.cat([uc_cross, uc_cross], dim=0)
            uc_full = {"c_concat": [c_cat], "c_crossattn": [uc_cross]}

            # if not do_full_sample:
            # # encode (scaled latent)
            #     z_enc = sampler.stochastic_encode(
            #         z, torch.tensor([t_enc] * opt.n_samples).to(model.device))
            # else:
            z_enc = torch.randn_like(z)
            
            z_enc = torch.cat([z_enc, z_enc], dim=0)
            # decode it
            samples = sampler.decode(z_enc, cond, t_enc, unconditional_guidance_scale=opt.scale,
                                        unconditional_conditioning=uc_full, 
                                        disparity=cc.squeeze(1),swapat=sa,shift_both=opt.shift_both,
                                        deblur=opt.deblur)

            x_samples = model.decode_first_stage(samples)
            x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
            output = rearrange_img(x_samples)
            output = rearrange(output,"b c h w -> c h (b w)")
            vutils.save_image(output, os.path.join(opt.outdir,'seed_%s.png'%opt.seed), normalize=True)

   


if __name__ == "__main__":
    opt = parse_args()
    main(opt)
