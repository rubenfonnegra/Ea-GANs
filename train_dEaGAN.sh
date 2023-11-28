python train.py --dataroot ./datasets/example_data \
                --name example_dEaGAN --model dea_gan \
                --which_model_netG unet_128 \
                --which_direction AtoB --lambda_A 300 \
                --dataset_mode aligned --use_dropout \
                --batchSize 6 --niter 100 \
                --niter_decay 50 --lambda_sobel 100 \
                --labelSmooth --rise_sobelLoss



tmux send-keys -t sp_01.0 " \
cd /home/ruben.fonnegra/workspaces/Ea-GANs 
source /home/ruben.fonnegra/venvs/sp_env/bin/activate 
python train.py --dataroot datasets/Duke/Duke_tiff/ \
                --inp_seq pre --out_seq post_1 --quality 3T \
                --name dEaGAN_1 --model dea_gan \
                --which_model_netG unet_128 \
                --which_direction AtoB --lambda_A 300 \
                --dataset_mode aligned --use_dropout \
                --batchSize 6 --niter 100 \
                --niter_decay 50 --lambda_sobel 100 \
                --labelSmooth --rise_sobelLoss \
                --fineSize 256" ENTER


tmux send-keys -t sp_01.0 " \
cd /home/ruben.fonnegra/workspaces/Ea-GANs 
source /home/ruben.fonnegra/venvs/sp_env/bin/activate 
python train.py --dataroot datasets/Duke/Duke_tiff/ \
                --inp_seq pre --out_seq post_1 --quality 1T \
                --name dEaGAN_1T1 --model dea_gan \
                --which_model_netG unet_128 \s
                --which_direction AtoB --lambda_A 300 \
                --dataset_mode aligned --use_dropout \
                --batchSize 6 --niter 100 \
                --niter_decay 50 --lambda_sobel 100 \
                --labelSmooth --rise_sobelLoss \
                --fineSize 256" ENTER
