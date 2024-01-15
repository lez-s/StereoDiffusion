
source ./venvStereoDiffusion/bin/activate

python ./StereoDiffusion/depth2stereoimg.py --ckpt=models/512-depth-ema.ckpt \
--depthmodel_path="models/dpt_hybrid-midas-501f0c75.pt" \
--n_samples=2 \
--direction="uni" \
--seed=2315 \
--deblur \
--prompt="An astronaut is riding horse,high quailty,photorealistic" \
--init_img="ori_img/horse.jpg" \
# --prompt="A GT car,high quailty,photorealistic," \
# --init_img="/home/lez/StereoDiffusion/ori_img/car.jpg" \


# --deblur