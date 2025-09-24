import numpy as np
import matplotlib.pyplot as plt
import time
import functools

import torch

from config import config
import measurement
import setup_utils

def get_mts_measurements(theta, update_fn, ctx_args):
    update_fn(theta)
    msr = measurement.Simulated(ctx_args['scene'])
    msr.calibrate(setup_utils.cal_path())
    return msr

def grad_of_gaussiankernel(x, sigma):
    grad_of_gauss = -(x / sigma ** 2) * calc_gauss(x, mu=0.0, sigma=sigma)
    return grad_of_gauss


def calc_gauss(x, mu=0.0, sigma=1.0):
    return 1.0 / (sigma * (2.0 * np.pi)**0.5) * torch.exp(-0.5 * ((x - mu) / sigma) ** 2)


def mc_estimate(f_xi, p_xi):
    N = f_xi.shape[0]
    estimate = 1. / N * (f_xi / p_xi).sum(dim=0)  # average along batch axis, leave dimension axis unchanged
    return estimate

def convolve(kernel_fn, render_fn, importance_fn, theta, nsamples, context_args, *args):
    # sample, get kernel(samples), get simulation(samples), return mc estimate of output
    # expect theta to be of shape [1, n], where n is dimensionality

    dim = theta.shape[-1]
    sigma = context_args['sigma']
    update_fn = context_args['update_fn']

    if context_args['sampler'] == 'uniform':
        raise NotImplementedError("for now only IS sampler supported")

    # get importance-sampled taus
    tau, pdf = importance_fn(nsamples, sigma, context_args['antithetic'], dim, context_args['device'])

    # get kernel weight at taus
    weights = kernel_fn(tau, sigma)

    # twice as many samples when antithetic
    if context_args['antithetic']:
        nsamples *= 2

    # shift samples around current parameter
    theta_p = torch.cat([theta] * nsamples, dim=0) - tau

    measurements = render_fn(theta_p, update_fn, context_args)    # output shape [N]

    # weight output by kernel, mc-estimate gradient
    output = measurements.unsqueeze(-1) * weights
    forward_output = mc_estimate(output, pdf)

    return forward_output

def importance_gradgauss(n_samples, sigma, is_antithetic, dim, device):
    eps = 0.00001
    randoms = torch.rand(n_samples, dim).to(device)

    def icdf(x, sigma):
        res = torch.zeros_like(x).to(device)
        res[mask == 1] = torch.sqrt(-2.0 * sigma ** 2 * torch.log(2.0 * (1.0 - x[mask == 1])))
        res[mask == -1] = torch.sqrt(-2.0 * sigma ** 2 * torch.log(2.0 * x[mask == -1]))
        return res

    # samples and AT samples
    if is_antithetic:
        randoms = torch.cat([randoms, 1.0 - randoms])

    # avoid NaNs bc of numerical instabilities in log
    randoms[torch.isclose(randoms, torch.ones_like(randoms))] -= eps
    randoms[torch.isclose(randoms, torch.zeros_like(randoms))] += eps
    randoms[torch.isclose(randoms, torch.full_like(randoms, fill_value=0.5))] += eps
    randoms[torch.isclose(randoms, torch.full_like(randoms, fill_value=-0.5))] -= eps

    mask = torch.where(randoms < 0.5, -1.0, 1.0)
    x_i = icdf(randoms, sigma=sigma) * mask

    f_xi = torch.abs(x_i) * (1.0 / sigma ** 2) * calc_gauss(x_i, mu=0.0, sigma=sigma)
    f_xi[f_xi == 0] += eps
    p_xi = 0.5 * sigma * (2.0 * np.pi)**0.5 * f_xi

    return x_i, p_xi

def smoothFn(func=None, context_args=None, device='cuda'):
    if func is None:
        return functools.partial(smoothFn, context_args=context_args, device=device)

    @functools.wraps(func)
    def wrapper(input_tensor, context_args, *args):
        class SmoothedFunc(torch.autograd.Function):

            @staticmethod
            def forward(ctx, input_tensor, context_args, *args):

                original_input_shape = input_tensor.shape
                importance_fn = importance_gradgauss

                forward_output = convolve(grad_of_gaussiankernel, func, importance_fn, input_tensor, context_args['nsamples'], context_args, args)

                # save for bw pass
                ctx.fw_out = forward_output
                ctx.original_input_shape = original_input_shape

                return forward_output.mean()

            @staticmethod
            def backward(ctx, dy):
                # Pull saved tensors
                original_input_shape = ctx.original_input_shape
                fw_out = ctx.fw_out
                grad_in_chain = dy * fw_out

                return grad_in_chain.reshape(original_input_shape), None

        return SmoothedFunc.apply(input_tensor, context_args, *args)

    return wrapper

def update_sigma_linear(it, sigma_0, sigma_min, n=400, const_first=100):
    return sigma_0 - (it - const_first) * (sigma_0 - sigma_min) / (n - const_first)


def run_scheduler_step(curr_sigma, curr_iter, sigma_initial, sigma_min, n, const_first_n, const_last_n=None):
    n_real = n - const_last_n if const_last_n else n
    newsigma = update_sigma_linear(curr_iter, sigma_initial, sigma_min, n_real, const_first_n)
    return newsigma

def sim_smooth(perturbed_theta, update_fn, ctx_args):
    # simulate with each perturbed position, get the final image, compute loss, batch, return
    # perturbed_thetas is expected to be of dim [nsamples, ndim]
    with torch.no_grad():
        losses = []
        for j in range(perturbed_theta.shape[0]):       # for each sample
            perturbed_msr = get_mts_measurements(perturbed_theta[j, :], update_fn, ctx_args)
            perturbed_loss = ctx_args['loss_fn'](perturbed_msr, ctx_args['gt_msr'])
            losses.append(perturbed_loss)

        loss = torch.stack(losses)
    return loss

def plt_errors(msr_err, param_err, title):
    #plt.plot(param_err, c='blue', label='Param. L1')
    plt.plot(msr_err, c='orange', label='L2 Hamm')
    plt.title(title)
    plt.legend()
    plt.show()

def run_optimization(hparams, optim, theta, gt_theta, ctx_args, schedule_fn, update_fn):
    sigma = hparams['sigma']

    theta_optim = theta
    min_loss = float('inf')

    init_msr = get_mts_measurements(theta, update_fn, ctx_args)

    # Set up smoothed simulation
    smooth_mts = smoothFn(sim_smooth, context_args=None, device=ctx_args['device'])

    msr_errors, param_errors = [], []
    msr_errors.append(ctx_args['loss_fn'](init_msr, ctx_args['gt_msr']).item())
    #param_errors.append(torch.abs(torch.norm(theta - gt_theta)).item())
    param_error = torch.abs(theta - gt_theta).tolist()
    param_error = [round(p, 6) for p in param_error]

    print(f"Running {hparams['epochs']} epochs with {hparams['nsamples']} samples and sigma={hparams['sigma']}")
    print(f"Iter 0/{hparams['epochs']}, ParamLoss: {param_error}, "
                  f"L2HammLoss: {msr_errors[-1]:.8f}")

    # --------------- run optimization
    for j in range(hparams['epochs']):
        start = time.time()
        optim.zero_grad()

        loss = smooth_mts(theta.unsqueeze(0), ctx_args)
        loss.backward()

        optim.step()

        # potential sigma scheduling:
        if j > hparams['anneal_const_first'] and hparams['sigma_annealing'] and sigma >= hparams['anneal_sigma_min']:
            sigma = schedule_fn(sigma, curr_iter=j + 1, n=hparams['epochs'],
                                sigma_initial=hparams['sigma'],
                                sigma_min=hparams['anneal_sigma_min'],
                                const_first_n=hparams['anneal_const_first'],
                                const_last_n=hparams['anneal_const_last'])
            ctx_args['sigma'] = sigma
        iter_time = time.time() - start

        # logging, timing, plotting, etc...
        with torch.no_grad():

            # calc loss btwn rendering with current parameter (non-blurred)
            msr_curr = get_mts_measurements(theta, update_fn, ctx_args)
            msr_errors.append(ctx_args['loss_fn'](msr_curr, ctx_args['gt_msr']).item())
            if(msr_errors[-1] < min_loss):
                theta_optim = theta.detach().clone()
                min_loss = msr_errors[-1]
                print(f"theta optim: {theta_optim}, min loss: {min_loss}")
            param_error = torch.abs(theta - gt_theta).tolist()
            param_error = [round(p, 6) for p in param_error]

            print(f"Iter {j + 1}/{hparams['epochs']}, ParamLoss: {param_error}, "
                  f"L2HammLoss: {msr_errors[-1]:.8f} - Time: {iter_time:.4f}")
        #if(min_loss < 1e-5):
        #    break

    plt_errors(msr_errors, param_errors, title=f'Final, after {hparams["epochs"]} iterations')
    print("Done.")
    return theta_optim

