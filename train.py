import time
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer

import sys


def main():
    #
    opt = TrainOptions().parse()
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)

    model = create_model(opt)
    visualizer = Visualizer(opt)

    total_steps = 0

    for epoch in range(1, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            total_steps += opt.batchSize
            epoch_iter = total_steps - dataset_size * (epoch - 1)
            model.set_input(data)  
            model.optimize_parameters()

            if total_steps % opt.display_freq == 0:
                visualizer.display_current_results(model.get_current_img(), epoch)

            if total_steps % opt.print_freq == 0:
                errors = model.get_current_errors()
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_errors(epoch, epoch_iter, errors, t)
                # if opt.display_id > 0:
                #     visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                    (epoch, total_steps))
                model.save('latest')

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                (epoch, total_steps))
            model.save('latest')
            model.save(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
            (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

        if opt.rise_sobelLoss and epoch <= 20:
            model.update_sobel_lambda(epoch)

        if epoch > opt.niter:
            model.update_learning_rate()


if __name__ == "__main__":
    #
    """ 
    param = sys.argv.append
    
    args = "--dataroot datasets/Duke/Duke_tiff/ \
            --inp_seq pre --out_seq post_1 --quality 3T \
            --name example_dEaGAN --model dea_gan \
            --which_model_netG unet_128 \
            --which_direction AtoB --lambda_A 300 \
            --dataset_mode aligned --use_dropout \
            --batchSize 6 --niter 100 \
            --niter_decay 50 --lambda_sobel 100 \
            --labelSmooth --rise_sobelLoss \
            --fineSize 256"
    
    for arg in args.split(" "): 
        if arg: param(arg)
    """
    main()