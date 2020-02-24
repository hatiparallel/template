import numpy as np
import torch


def get_output_directory(args):
    output_directory = os.path.join('results',
                                    '{}.arch={}.criterion={}.lr={}.momentum={}.weightdecay={}.bs={}'.
                                    format(args.data, args.arch, args.criterion, args.lr, args.momentum, args.weight_decay, args.batch_size, args))

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    return output_directory


def save_checkpoint(state, is_best, epoch, output_directory):
    checkpoint_filename = os.path.join(
        output_directory, 'checkpoint-' + str(epoch) + '.pth.tar')
    torch.save(state, checkpoint_filename)
    if is_best:
        best_filename = os.path.join(output_directory, 'model_best.pth.tar')
        shutil.copyfile(checkpoint_filename, best_filename)
    if epoch > 0:
        prev_checkpoint_filename = os.path.join(
            output_directory, 'checkpoint-' + str(epoch - 1) + '.pth.tar')
        if os.path.exists(prev_checkpoint_filename):
            os.remove(prev_checkpoint_filename)