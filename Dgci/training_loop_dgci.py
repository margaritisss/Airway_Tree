import torch
import time
import numpy as np
import os
import utils
import json
import shutil
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
from collections import OrderedDict
from torch.nn import parameter


def run_validation(model, val_dataloader, loss_schedules, total_steps):
    model.eval()
    comp_sums, loss_sum, steps = OrderedDict(), 0.0, 0
    with torch.enable_grad():                       # required: forward() calls autograd.grad
        for model_input, gt in val_dataloader:
            model_input = {k: v.cuda() for k, v in model_input.items()}
            gt          = {k: v.cuda() for k, v in gt.items()}
            losses = model(model_input, gt)
            total  = 0.
            for name, loss in losses.items():
                sl = loss.mean()
                if loss_schedules is not None and name in loss_schedules:
                    sl = sl * loss_schedules[name](total_steps)
                total = total + sl
                comp_sums[name] = comp_sums.get(name, 0.0) + float(sl)
            loss_sum += float(total)
            steps    += 1
            del losses, total                       # frees the second-order graph
    model.train()
    optim_zeroed = None                             # no .grad is populated; nothing to clear
    return loss_sum, comp_sums, steps

def save_checkpoint(state_dict, path):
    """Atomic save: writing to a temp file, then renaming."""
    tmp = path + '.tmp'
    torch.save(state_dict, tmp)
    os.replace(tmp, path)

# training_loop_dgci.py
def train(model, train_dataloader, epochs, lr, steps_til_summary, epochs_til_checkpoint, model_dir, val_dataloader=None, loss_schedules=None, is_train=True, optim='Adam', **kwargs):
    print('Training Info:')
    print('num_instances:\t\t', kwargs['num_instances'])
    print('batch_size:\t\t', kwargs['batch_size'])
    print('epochs:\t\t\t', epochs)
    print('learning rate:\t\t', lr)

    # -------------    Logging Loss Arguments     ---------------------
    for key in kwargs:                                                 # Iterates through all the extra keyword arguments (kwargs) passed into the train function.
        if 'loss' in key:
            print(key + ':\t', kwargs[key])                            # it prints the name of the key and its corresponding value to the console. This is useful for logging which loss functions or loss weights are being applied during the run.

    # ----------     Optimizer Setup & Parameter Filtering    -----------------
    start_epoch  = 0
    resume_steps = 0
    
    if is_train:                                                       # Ensures the optimizer is only initialized if the script is explicitly set to training mode.
        if optim == 'Adam':
            for name, param in model.module.named_parameters():        # Iterates through all the parameters of the model, retrieving both the name and the parameter tensor itself.
                if not name.find('detach') == -1:                      # Checks if the substring 'detach' is present in the parameter's name. If it is, the parameter is skipped and not included in the optimizer's parameter list. This is useful for excluding certain parameters from training, such as those that should remain fixed or are part of a pre-trained model.
                    print(name)
            optim = torch.optim.Adam(lr=lr,params=[param for name, param in model.module.named_parameters() if  name.find('detach') == -1])  # Initializes the Adam optimizer with the specified learning rate (lr) and a filtered list of model parameters. Only parameters whose names do not contain 'detach' are included in the optimizer's parameter list, ensuring that only the desired parameters are updated during training.

    # -----------------     Resuming Optimizer State from Checkpoint    -----------------
            if 'checkpoint_path' in kwargs and len(kwargs['checkpoint_path']) > 0:
                state_dict = torch.load(kwargs['checkpoint_path'].replace('model', 'optim'))
                optim.load_state_dict(state_dict)

                state_file = os.path.join(os.path.dirname(kwargs['checkpoint_path']), 'train_state.json')
                if os.path.exists(state_file):
                    with open(state_file) as f:
                        _st = json.load(f)
                    start_epoch  = _st['epoch']
                    resume_steps = _st['total_steps']
                    print('resuming from epoch %d (global step %d)' % (start_epoch, resume_steps))
                else:
                    print('WARNING: no train_state.json next to the checkpoint - restarting epoch count at 0')

    # -----------------    Creating Output Directories     -----------------
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    summaries_dir = os.path.join(model_dir, 'summaries')
    utils.cond_mkdir(summaries_dir) 

    checkpoints_dir = os.path.join(model_dir, 'checkpoints')             # os.path.join is a command that joins two paths together. In this case, it joins the model_dir and 'checkpoints' to create a new path for the checkpoints directory.
    utils.cond_mkdir(checkpoints_dir)

    # -----------------    Initialization and Progress Tracking     -----------------
    writer      = SummaryWriter(summaries_dir)                           # Creates a SummaryWriter object that will write logs to the specified summaries_dir. This object is used to log various metrics and visualizations during training, which can later be viewed in TensorBoard.
    total_steps = resume_steps
    with tqdm(total=len(train_dataloader) * (epochs - start_epoch)) as pbar:
        train_losses = []

    # -----------------    Training Loop - Epoch Loop & Epoch Checkpointing    -----------------
        for epoch in range(start_epoch, epochs):
            if not epoch % epochs_til_checkpoint:
                if is_train:
                    model_path = os.path.join(checkpoints_dir, 'model_latest.pth')
                    optim_path = os.path.join(checkpoints_dir, 'optim_latest.pth')

                    save_checkpoint(model.module.state_dict(), model_path)
                    save_checkpoint(optim.state_dict(),        optim_path)

                    with open(os.path.join(checkpoints_dir, 'train_state.json'), 'w') as f:
                        json.dump({'epoch': epoch, 'total_steps': total_steps}, f)

                    keep_every = kwargs.get('keep_every', 100)
                    if keep_every and not epoch % keep_every:
                        shutil.copyfile(model_path, os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % epoch))
                        shutil.copyfile(optim_path, os.path.join(checkpoints_dir, 'optim_epoch_%04d.pth' % epoch))

                np.savetxt(os.path.join(checkpoints_dir, 'train_losses_latest.txt'), np.array(train_losses))

    # -----------------    Inner Batch Loop & Data Preparation    -----------------  
            epoch_loss_sum       = 0.0
            epoch_component_sums = OrderedDict()
            epoch_steps          = 0
            epoch_start_time     = time.time() 

            for step, (model_input, gt) in enumerate(train_dataloader):
                start_time  = time.time() 
                model_input = {key: value.cuda() for key, value in model_input.items()}
                gt          = {key: value.cuda() for key, value in gt.items()}

    # -----------------    Forward Pass & Loss Calculation   -----------------
                if is_train:
                    losses = model(model_input, gt)  # this 

                train_loss    = 0.
                output_string = "" 
                
                for loss_name, loss in losses.items(): 
                    single_loss = loss.mean()
                    if loss_schedules is not None and loss_name in loss_schedules:
                        writer.add_scalar(loss_name + "_weight", loss_schedules[loss_name](total_steps), total_steps)
                        single_loss *= loss_schedules[loss_name](total_steps) 

                    writer.add_scalar(loss_name, single_loss, total_steps)
                    train_loss    += single_loss
                    output_string += "%s %.3f " % (loss_name, float(single_loss))
                    epoch_component_sums[loss_name] = epoch_component_sums.get(loss_name, 0.0) + float(single_loss)

    # -----------------    Logging & Frequent Saving    -----------------  
              
                # if total_steps % 10 == 0:
                #     tqdm.write(output_string)                                                 # this prints the output_string to the console every 10 steps, providing a summary of the current loss values for each loss component. This helps in monitoring the training progress and identifying any potential issues with specific loss terms.
                assert not torch.isnan(train_loss), 'NaN loss at step %d' % total_steps       # This line checks if the computed train_loss is NaN (Not a Number) at the current step. If it is, an assertion error is raised with a message indicating the step number where the NaN loss occurred. This is a safeguard to catch numerical instability or issues in the training process early on.

                train_losses.append(train_loss.item())  
                writer.add_scalar("total_train_loss", train_loss, total_steps)
                epoch_loss_sum += train_losses[-1]
                epoch_steps    += 1

                # if not total_steps % steps_til_summary:
                #     if is_train:
                #         torch.save(model.module.state_dict(), os.path.join(checkpoints_dir, 'model_current.pth'))

    # -----------------    Backpropagation & Optimization    -----------------

                optim.zero_grad()       # this line resets the gradients of all model parameters to zero before computing the gradients for the current batch. In PyTorch, gradients are accumulated by default, so if you don't zero them out, the gradients from previous batches would be added to the current gradients, leading to incorrect updates.
                train_loss.backward()   # this line computes the gradients of the loss with respect to the model parameters by performing backpropagation. The backward() function calculates the derivative of the loss function with respect to each parameter (weight and bias) in the model, allowing for gradient-based optimization.
                optim.step()            # this line updates the model parameters based on the computed gradients. The step() function of the optimizer applies the optimization algorithm (in this case, Adam) to adjust the parameters in the direction that minimizes the loss, effectively performing a single optimization step.

    # -----------------    Step Updates & Final Saving    -----------------

                pbar.update(1)  
                # if not total_steps % steps_til_summary: 
                #     tqdm.write("Epoch %d, Total loss %0.6f, iteration time %0.6f" % (epoch, train_loss, time.time() - start_time))
                total_steps += 1
            
            if epoch_steps > 0:
                epoch_mean_loss = epoch_loss_sum / epoch_steps
                epoch_comp_str  = " ".join("%s %.3f" % (name, s / epoch_steps)
                                           for name, s in epoch_component_sums.items())

                tqdm.write("=" * 100)
                tqdm.write("EPOCH %d DONE | steps %d | mean total loss %0.6f | sum total loss %0.3f | epoch time %0.1fs"
                           % (epoch, epoch_steps, epoch_mean_loss, epoch_loss_sum, time.time() - epoch_start_time))
                tqdm.write("EPOCH %d MEAN COMPONENTS | %s" % (epoch, epoch_comp_str))

                writer.add_scalar("epoch/mean_total_loss", epoch_mean_loss, epoch)
                for name, s in epoch_component_sums.items():
                    writer.add_scalar("epoch/" + name, s / epoch_steps, epoch)

            epochs_til_val = kwargs.get('epochs_til_val', 1)
            if val_dataloader is not None and not epoch % epochs_til_val:
                v_loss_sum, v_comp_sums, v_steps = run_validation(model, val_dataloader,
                                                                  loss_schedules, total_steps)
                if v_steps > 0:
                    v_mean     = v_loss_sum / v_steps
                    v_comp_str = " ".join("%s %.3f" % (name, s / v_steps)
                                          for name, s in v_comp_sums.items())
                    tqdm.write("VALIDATION EPOCH %d | steps %d | mean total loss %0.6f" % (epoch, v_steps, v_mean))
                    tqdm.write("VALIDATION EPOCH %d MEAN COMPONENTS | %s" % (epoch, v_comp_str))
                    writer.add_scalar("val/mean_total_loss", v_mean, epoch)
                    for name, s in v_comp_sums.items():
                        writer.add_scalar("val/" + name, s / v_steps, epoch)
                else:
                    tqdm.write("VALIDATION EPOCH %d | skipped: val loader produced 0 batches" % epoch)
            tqdm.write("=" * 100)

        if is_train:
            torch.save(model.module.cpu().state_dict(), os.path.join(checkpoints_dir, 'model_final.pth'))
            torch.save(optim.state_dict(), os.path.join(checkpoints_dir, 'optim_final.pth'))
        np.savetxt(os.path.join(checkpoints_dir, 'train_losses_final.txt'), np.array(train_losses))
