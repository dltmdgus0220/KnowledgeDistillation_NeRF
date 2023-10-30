# python run_nerf_odist.py --config config_fern.txt
# tensorboard --logdir==
# 카메라는 real world -> nomalized plane -> image plane 이 목적
# nerf는 image plane -> nomalized plane -> real world 
# 수정 : 234, 261, 869, 925, 974, 973

import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
import tensorflow as tf
import numpy as np
import imageio
import json
import random
import time
from run_nerf_helpers import *
# from run_convnerf_helpers import *
# from run_mobilenerf5_helpers import *
# from run_unerf2_helpers import *
from load_llff import load_llff_data
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data

tf.compat.v1.enable_eager_execution() # session을 안쓰고 바로 계산 및 확인을 할 수 있도록

# print("=========================")
# from tensorflow.python.client import device_lib
# device_lib.list_local_devices()
# print("=========================")

def batchify(fn, chunk): # 15
    """Constructs a version of 'fn' that applies to smaller batches."""
    if chunk is None:
        return fn

    def ret(inputs): # input (65536, 90)
        return tf.concat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0) # output (65536, 4) (0~16383, 90) (16384~32767, 90) (32768~49151, 90) (49152~65335, 90) 이렇게 4개의 배치로 쪼개서 모델을 학습시키고 나온 output을 concat하여 최종 output으로 만듬
    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64): # netchunk 메모리 오류로 인해 512*32=16384로 낮춤, 원래는 full batch로 학습
    """Prepares inputs and applies network 'fn'."""
    # 14
    inputs_flat = tf.reshape(inputs, [-1, inputs.shape[-1]]) # inputs (1024, 64, 3), inputs_flat (65536, 3) ==> 1024*64=65536

    embedded = embed_fn(inputs_flat) # embedded (65536, 63) ==> positional encoding을 통해 60차원 늘려줌, L=10
    if viewdirs is not None:
        input_dirs = tf.broadcast_to(viewdirs[:, None], inputs.shape)
        input_dirs_flat = tf.reshape(input_dirs, [-1, input_dirs.shape[-1]]) # inputs_dirs (1024, 64, 3), inputs_dirs_flat (65536, 3) ==> 1024*64=65536, direction에 대한 input은 좌표에 대한 input과 shape이 동일함. direction은 원래 2차원이지만 z=1인 3차원으로 input에 넣어줌
        embedded_dirs = embeddirs_fn(input_dirs_flat) # embedded_dirs (65536, 27) ==> positional encoding을 통해 24차원 늘려줌, L=4
        embedded = tf.concat([embedded, embedded_dirs], -1) # embedded (65536, 90) ==> 63+27=90

    outputs_flat = batchify(fn, netchunk)(embedded) # outputs_flat (65536, 4)
    outputs = tf.reshape(outputs_flat, list(
        inputs.shape[:-1]) + [outputs_flat.shape[-1]])

    # print("outputs : ", outputs)
    return outputs # outputs (1024, 64, 4)



def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                isTrain=True,
                verbose=False):

    # 12

    """Volumetric rendering.

    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.

    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """

    def raw2outputs(raw, z_vals, rays_d): # 16
        """Transforms model's predictions to semantically meaningful values.

        Args:
          raw: [num_rays, num_samples along ray, 4]. Prediction from model.
          z_vals: [num_rays, num_samples along ray]. Integration time.
          rays_d: [num_rays, 3]. Direction of each ray.

        Returns:
          rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
          disp_map: [num_rays]. Disparity map. Inverse of depth map.
          acc_map: [num_rays]. Sum of weights along each ray.
          weights: [num_rays, num_samples]. Weights assigned to each sampled color.
          depth_map: [num_rays]. Estimated distance to object.
        """
        # Function for computing density from model prediction. This value is
        # strictly between [0, 1].
        def raw2alpha(raw, dists, act_fn=tf.nn.relu): return 1.0 - \
            tf.exp(-act_fn(raw) * dists) # 1 - exp(-volume_density * distance), distance : ti+1 좌표와 ti 좌표 사이 간격

        # Compute 'distance' (in time) between each integration time along a ray.
        dists = z_vals[..., 1:] - z_vals[..., :-1] # distance : ti+1 좌표와 ti 좌표 사이 간격

        # The 'distance' from the last integration time is infinity.
        dists = tf.concat(
            [dists, tf.broadcast_to([1e10], dists[..., :1].shape)],
            axis=-1)  # [N_rays, N_samples]

        # Multiply each distance by the norm of its corresponding direction ray
        # to convert to real world distance (accounts for non-unit directions).
        dists = dists * tf.linalg.norm(rays_d[..., None, :], axis=-1)

        # Extract RGB of each sample position along each ray.
        rgb = tf.math.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]

        # Add noise to model's predictions for density. Can be used to 
        # regularize network during training (prevents floater artifacts).
        noise = 0.
        if raw_noise_std > 0.:
            noise = tf.random.normal(raw[..., 3].shape) * raw_noise_std

        # Predict density of each sample along each ray. Higher values imply
        # higher likelihood of being absorbed at this point.
        alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]

        # Compute weight for RGB of each sample along each ray.  A cumprod() is
        # used to express the idea of the ray not having reflected up to this
        # sample yet.
        # [N_rays, N_samples]
        weights = alpha * \
            tf.math.cumprod(1.-alpha + 1e-10, axis=-1, exclusive=True) # [a, b, c] => [a ,a*b, a*b*c], T_i

        # Computed weighted color of each sample along each ray.
        rgb_map = tf.reduce_sum(
            weights[..., None] * rgb, axis=-2)  # [N_rays, 3]

        # Estimated depth map is expected distance.
        depth_map = tf.reduce_sum(weights * z_vals, axis=-1)

        # Disparity map is inverse depth.
        disp_map = 1./tf.maximum(1e-10, depth_map /
                                 tf.reduce_sum(weights, axis=-1))

        # Sum of weights along each ray. This value is in [0, 1] up to numerical error.
        acc_map = tf.reduce_sum(weights, -1)

        # To composite onto a white background, use the accumulated alpha map.
        if white_bkgd:
            rgb_map = rgb_map + (1.-acc_map[..., None])

        return rgb_map, disp_map, acc_map, weights, depth_map

    ###############################
    # batch size
    N_rays = ray_batch.shape[0] # N_rays = 1024, ray_batch [1024, 11] ==> [batchsize, rays_o + rays_d + near + far + viewdirs]

    # Extract ray origin, direction.
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each

    # Extract unit-normalized viewing direction.
    viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None

    # Extract lower, upper bound for ray distance.
    bounds = tf.reshape(ray_batch[..., 6:8], [-1, 1, 2])
    near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]

    # Decide where to sample along each ray. Under the logic, all rays will be sampled at
    # the same times.
    # Coarse(Stratified) Sampling 시작
    t_vals = tf.linspace(0., 1., N_samples) # N_samples = 64, 0에서 1 즉 near에서 far까지 64개 구간으로 나눔
    if not lindisp: # lindisp == False, not lindisp == True
        # Space integration times linearly between 'near' and 'far'. Same
        # integration points will be used for all rays.
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        # Sample linearly in inverse depth (disparity).
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))
    z_vals = tf.broadcast_to(z_vals, [N_rays, N_samples])

    # Perturb sampling time along each ray.
    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = tf.concat([mids, z_vals[..., -1:]], -1)
        lower = tf.concat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = tf.random.uniform(z_vals.shape)
        z_vals = lower + (upper - lower) * t_rand

    # Points in space to evaluate model at.
    pts = rays_o[..., None, :] + rays_d[..., None, :] * \
        z_vals[..., :, None]  # [N_rays, N_samples, 3], (1024,1,3) + (1024,1,3) * (1024,64,1) = (1024,64,3) ==> 3d voxel 좌표 = ray origin + ray direction * z
    # Coarse(Stratified) Sampling 끝
    # Evaluate model at each point.
    if isTrain == True: # train/val 할 때는 coarse+fine
        # Coarse network 통과
        raw = network_query_fn(pts, viewdirs, network_fn)  # [N_rays, N_samples, 4]
        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
            raw, z_vals, rays_d)
        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map
    
    # test/video 할 때는 fine만
    # Fine network 통과
    run_fn = network_fn if network_fine is None else network_fine
    raw = network_query_fn(pts, viewdirs, run_fn)
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
        raw, z_vals, rays_d)

    ret = {'rgb_map': rgb_map, 'disp_map': disp_map, 'acc_map': acc_map}
    if retraw:
        ret['raw'] = raw
    if isTrain == True:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        # ret['z_std'] = tf.math.reduce_std(z_samples, -1)  # [N_rays]

    for k in ret:
        tf.debugging.check_numerics(ret[k], 'output {}'.format(k))

    return ret


def batchify_rays(rays_flat, chunk=1024*32, **kwargs):
    # 11
    """Render rays in smaller minibatches to avoid OOM."""
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk): # chunk만큼 건너뛰면서 minibatch 방식으로 동작
        ret = render_rays(rays_flat[i:i+chunk], **kwargs) # rgb_map (1024,3), disp_map (1024,), acc_map (1024,), raw (1024, 128, 4), rgb0, disp0, acc0 이 세개는 fine network를 통과하기 전 coarse network만 통과했을때의 map들, z_std (1024,)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k: tf.concat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(H, W, focal,
           chunk=1024*32, rays=None, c2w=None, ndc=True,
           near=0., far=1.,
           use_viewdirs=False, c2w_staticcam=None,
           **kwargs):
    # 10
    """Render rays

    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.

    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """

    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, focal, c2w) # (378, 504, 3)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, focal, c2w_staticcam)

        # Make all directions unit magnitude. ==> 특정 direction으로 향하는 magnitude가 1인 단위 백터
        # shape: [batch_size, 3]
        viewdirs = viewdirs / tf.linalg.norm(viewdirs, axis=-1, keepdims=True)
        viewdirs = tf.cast(tf.reshape(viewdirs, [-1, 3]), dtype=tf.float32)

    sh = rays_d.shape  # [..., 3]
    if ndc: # nomalized device coordinate에 projection 시키면 정육면체 공간에서의 ray를 구할 수 있음, forward facing scene 같은 경우 360 데이터셋과 달리 사다리꼴 형태의 ray를 얻게된다.
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(
            H, W, focal, tf.cast(1., tf.float32), rays_o, rays_d)
    # rays_o (1024, 3), rays_d (1024, 3), viewdirs (1024, 3)

    # Create ray batch
    rays_o = tf.cast(tf.reshape(rays_o, [-1, 3]), dtype=tf.float32)
    rays_d = tf.cast(tf.reshape(rays_d, [-1, 3]), dtype=tf.float32)
    near, far = near * \
        tf.ones_like(rays_d[..., :1]), far * tf.ones_like(rays_d[..., :1]) # near 0으로 채워진 (1024,1), far 1로 채워진 (1024,1)

    # (ray origin, ray direction, min dist, max dist) for each ray
    rays = tf.concat([rays_o, rays_d, near, far], axis=-1) # (1024, 8) ==> 3+3+1+1
    if use_viewdirs:
        # (ray origin, ray direction, min dist, max dist, normalized viewing direction)
        rays = tf.concat([rays, viewdirs], axis=-1) # (1024, 11) ==> 8+3
    # Render and reshape
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = tf.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_path(render_poses, hwf, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):

    H, W, focal = hwf

    if render_factor != 0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    disps = []

    t = time.time()
    for i, c2w in enumerate(render_poses):
        print(i, time.time() - t)
        t = time.time()
        rgb, disp, acc, _ = render(
            H, W, focal, chunk=chunk, c2w=c2w[:3, :4], isTrain = False, **render_kwargs)
        rgbs.append(rgb.numpy())
        disps.append(disp.numpy())
        if i == 0:
            print(rgb.shape, disp.shape)

        if gt_imgs is not None and render_factor == 0:
            p = -10. * np.log10(np.mean(np.square(rgb - gt_imgs[i])))
            print(p)

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps


def create_nerf(args): # 6
    """Instantiate NeRF's MLP model."""
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed) # positional embedding, multires 10, i_embed 0

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(
            args.multires_views, args.i_embed)
    output_ch = 4 # density 1 + rgb 3
    skips = [4]
    model = init_nerf_model(
        D=args.netdepth, W=args.netwidth,
        input_ch=input_ch, output_ch=output_ch, skips=skips,
        input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs) # model 정의
    grad_vars_teacher = model.trainable_variables # 모델 파라미터, (63,256) weight와 256 bias -> (256,256) weight와 256 bias * 6, skips 일때는 (256+63,256) weight와 256 bias -> density 추출을 위한 (256,1) weight -> (256+27,128) weight와 bias 128 -> (128,3) weight와 bias 3
    models = {'model': model}
    print(model.summary())

    model_fine = None

    # fine network에서는 positional encoding X
    model_fine = init_nerf_model(
        D=args.netdepth_fine, W=args.netwidth_fine,
        input_ch=input_ch, output_ch=output_ch, skips=[4],
        input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs) # fine model 정의
    grad_vars_student = model_fine.trainable_variables # fine network까지 총 두 번에 네트워크를 거쳐야하기 때문에 업데이트할 모델 파라미터를 두배 해줌
    models['model_fine'] = model_fine
    print(model_fine.summary())

    def network_query_fn(inputs, viewdirs, network_fn): # 13
        return run_network(
        inputs, viewdirs, network_fn,
        embed_fn=embed_fn,
        embeddirs_fn=embeddirs_fn,
        netchunk=args.netchunk)

    render_kwargs_train = {
        'network_query_fn': network_query_fn,
        'perturb': args.perturb,
        'N_importance': args.N_importance,
        'network_fine': model_fine,
        'N_samples': args.N_samples,
        'network_fn': model,
        'use_viewdirs': args.use_viewdirs,
        'white_bkgd': args.white_bkgd,
        'raw_noise_std': args.raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {
        k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    start = 0
    basedir = args.basedir
    expname = args.expname

    if args.ft_path is not None and args.ft_path != 'None': # 저장된 weight 불러오는 코드
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if
                 ('model_' in f and 'fine' not in f and 'optimizer' not in f)]
    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ft_weights = ckpts[-1]
        print('Reloading from', ft_weights)
        model.set_weights(np.load(ft_weights, allow_pickle=True))
        start = int(ft_weights[-10:-4]) + 1
        print('Resetting step to', start)

        if model_fine is not None:
            ft_weights_fine = '{}_fine_{}'.format(
                ft_weights[:-11], ft_weights[-10:])
            print('Reloading fine from', ft_weights_fine)
            model_fine.set_weights(np.load(ft_weights_fine, allow_pickle=True))

    return render_kwargs_train, render_kwargs_test, start, grad_vars_teacher, grad_vars_student, models


def config_parser(): # 2
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--expname", type=str, help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/',
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str,
                        default='./data/llff/fern', help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int,
                        default=8, help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=64,
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float,
                        default=5e-4, help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250,
                        help='exponential learning rate decay (in 1000s)')
    parser.add_argument("--chunk", type=int, default=1024*16,
                        help='number of rays processed in parallel, decrease if running out of memory') # before 1024*32
    parser.add_argument("--netchunk", type=int, default=1024*32,
                        help='number of pts sent through network in parallel, decrease if running out of memory') # before 1024*64
    parser.add_argument("--no_batching", action='store_true',
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--random_seed", type=int, default=None,
                        help='fix random seed for repeatability')
    
    # pre-crop options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops')    

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true',
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0,
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0.,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true',
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff',
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8,
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    # deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek',
                        help='options : armchair / cube / greek / vase')

    # blender flags
    parser.add_argument("--white_bkgd", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true',
                        help='load blender synthetic data at 400x400 instead of 800x800')

    # llff flags
    parser.add_argument("--factor", type=int, default=8,
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true',
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true',
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true',
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8,
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100,
                        help='frequency of console printout and metric loggin') # before 100
    parser.add_argument("--i_img",     type=int, default=500,
                        help='frequency of tensorboard image logging') # before 500
    parser.add_argument("--i_weights", type=int, default=10000,
                        help='frequency of weight ckpt saving') # before 10000
    parser.add_argument("--i_testset", type=int, default=100000,
                        help='frequency of testset saving') # before 50000
    parser.add_argument("--i_video",   type=int, default=100000,
                        help='frequency of render_poses video saving') # before 50000

    return parser

def train(): # 1
    global val_loss_psnr
    global iter_time
    val_loss_psnr = []
    iter_time= []

    parser = config_parser()
    args = parser.parse_args()
    
    if args.random_seed is not None:
        print('Fixing random seed', args.random_seed)
        np.random.seed(args.random_seed)
        tf.compat.v1.set_random_seed(args.random_seed)

    # Load data
    # 3
    if args.dataset_type == 'llff':
        images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)
        # print("==============================")
        # print(poses.shape, poses)
        hwf = poses[0, :3, -1]
        poses = poses[:, :3, :4]
        # print("==============================")
        # print(poses.shape, poses)
        # print("==============================")
        # print(hwf.shape, hwf)
        # print("==============================")
        # print(images.shape, render_poses.shape, render_poses)
        print('Loaded llff', images.shape,
              render_poses.shape, hwf, args.datadir)
        
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                            (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = tf.reduce_min(bds) * .9
            far = tf.reduce_max(bds) * 1.
        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)
        print(i_train, i_val, i_test)

    elif args.dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split = load_blender_data(
            args.datadir, args.half_res, args.testskip)
        print('Loaded blender', images.shape,
              render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near = 2.
        far = 6.

        if args.white_bkgd:
            images = images[..., :3]*images[..., -1:] + (1.-images[..., -1:])
        else:
            images = images[..., :3]

    elif args.dataset_type == 'deepvoxels':

        images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.shape,
                                                                 basedir=args.datadir,
                                                                 testskip=args.testskip)

        print('Loaded deepvoxels', images.shape,
              render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:, :3, -1], axis=-1))
        near = hemi_R-1.
        far = hemi_R+1.

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # Cast intrinsics to right types
    # 4
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if args.render_test: # false
        render_poses = np.array(poses[i_test])

    # Create log dir and copy the config file
    # 5
    basedir = args.basedir # ./logs
    expname = args.expname # fern_test
    os.makedirs(os.path.join(basedir, expname), exist_ok=True) # ./logs/fern_test
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    # 6
    render_kwargs_train, render_kwargs_test, start, grad_vars_teacher, grad_vars_student, models = create_nerf(
        args)

    bds_dict = {
        'near': tf.cast(near, tf.float32),
        'far': tf.cast(far, tf.float32),
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Short circuit if only rendering out from trained model
    if args.render_only: # false
        print('RENDER ONLY')
        if args.render_test:
            # render_test switches to test poses
            images = images[i_test]
        else:
            # Default is smoother render_poses path
            images = None

        testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format(
            'test' if args.render_test else 'path', start))
        os.makedirs(testsavedir, exist_ok=True)
        print('test poses shape', render_poses.shape)

        rgbs, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test,
                              gt_imgs=images, savedir=testsavedir, render_factor=args.render_factor)
        print('Done rendering', testsavedir)
        imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'),
                         to8b(rgbs), fps=30, quality=8)

        return

    # Create optimizer
    # 7
    lrate = args.lrate # 0.0005
    if args.lrate_decay > 0: # 250
        lrate = tf.keras.optimizers.schedules.ExponentialDecay(lrate,
                                                               decay_steps=args.lrate_decay * 1000, decay_rate=0.1)
    optimizer = tf.keras.optimizers.Adam(lrate)
    models['optimizer'] = optimizer

    global_step = tf.compat.v1.train.get_or_create_global_step()
    global_step.assign(start)

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand # 1024
    use_batching = not args.no_batching # no_batching = False, use_batching = True
    if use_batching:
        # For random ray batching.
        #
        # Constructs an array 'rays_rgb' of shape [N*H*W, 3, 3] where axis=1 is
        # interpreted as,
        #   axis=0: ray origin in world space
        #   axis=1: ray direction in world space
        #   axis=2: observed RGB color of pixel
        print('get rays')
        # get_rays_np() returns rays_origin=[H, W, 3], rays_direction=[H, W, 3]
        # for each pixel in the image. This stack() adds a new dimension.
        rays = [get_rays_np(H, W, focal, p) for p in poses[:, :3, :4]]
        rays = np.stack(rays, axis=0)  # [N, ro+rd, H, W, 3] ==> [20, 2, 378, 504, 3]
        print('done, concats')
        rays_rgb = np.concatenate([rays, images[:, None, ...]], 1) # [N, ro+rd+rgb, H, W, 3] ==> [20, 3, 378, 504, 3]
        rays_rgb = np.transpose(rays_rgb, [0, 2, 3, 1, 4]) # [N, H, W, ro+rd+rgb, 3] ==> [20, 378, 504, 3, 3] 순서바꿔줌
        rays_rgb = np.stack([rays_rgb[i]
                             for i in i_train], axis=0)  # train images only [17, 378, 504, 3, 3]
        rays_rgb = np.reshape(rays_rgb, [-1, 3, 3]) # [(N-1)*H*W, ro+rd+rgb, 3] ==> [17*378*504=3238704, 3, 3]
        rays_rgb = rays_rgb.astype(np.float32)
        print('shuffle rays')
        np.random.shuffle(rays_rgb)
        print('done')
        i_batch = 0

    N_iters = 100001
    print('Begin')
    print(N_iters)
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    # Summary writers
    try:
        os.makedirs(os.path.join(basedir, 'summaries', expname), exist_ok=True)
    except:
        print()
    writer = tf.contrib.summary.create_file_writer(
    # writer = tf.summary.create_file_writer(
        os.path.join(basedir, 'summaries', expname))
    writer.set_as_default()

    # 8
    for i in range(start, N_iters):
        time0 = time.time()

        # Sample random ray batch
        if use_batching:
            # Random over all images
            batch = rays_rgb[i_batch:i_batch+N_rand]  # [B, 2+1, 3*?] , rays_rgb (3238704, 3, 3), batch (1024, 3*, 3), 17*378*504=3238704 ==> train set 17장에 대한 matrix
            batch = tf.transpose(batch, [1, 0, 2]) # batch (3*, 1024, 3)

            # batch_rays[i, n, xyz] = ray origin or direction, example_id, 3D position 
            # target_s[n, rgb] = example_id, observed color.
            batch_rays, target_s = batch[:2], batch[2] # batch_rays (2, 1024, 3), batch (1024, 3), batch_rays에 할당되지 않은 나머지 1개가 batch로 할당

            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                np.random.shuffle(rays_rgb)
                i_batch = 0

        else: # batch를 사용하기 때문에 들어가지않음
            # Random from one image
            img_i = np.random.choice(i_train)
            target = images[img_i]
            pose = poses[img_i, :3, :4]

            if N_rand is not None:
                rays_o, rays_d = get_rays(H, W, focal, pose)
                if i < args.precrop_iters:
                    dH = int(H//2 * args.precrop_frac)
                    dW = int(W//2 * args.precrop_frac)
                    coords = tf.stack(tf.meshgrid(
                        tf.range(H//2 - dH, H//2 + dH), 
                        tf.range(W//2 - dW, W//2 + dW), 
                        indexing='ij'), -1)
                    if i < 10:
                        print('precrop', dH, dW, coords[0,0], coords[-1,-1])
                else:
                    coords = tf.stack(tf.meshgrid(
                        tf.range(H), tf.range(W), indexing='ij'), -1)
                coords = tf.reshape(coords, [-1, 2])
                select_inds = np.random.choice(
                    coords.shape[0], size=[N_rand], replace=False)
                select_inds = tf.gather_nd(coords, select_inds[:, tf.newaxis])
                rays_o = tf.gather_nd(rays_o, select_inds)
                rays_d = tf.gather_nd(rays_d, select_inds)
                batch_rays = tf.stack([rays_o, rays_d], 0)
                target_s = tf.gather_nd(target, select_inds)

        #####  Core optimization loop  #####
        # 9
        with tf.GradientTape(persistent = True) as tape:

            # Make predictions for color, disparity, accumulated opacity.
            rgb, disp, acc, extras = render(
                H, W, focal, chunk=args.chunk, rays=batch_rays,
                verbose=i < 10, retraw=True, **render_kwargs_train)
            
            trans = extras['raw'][..., -1]

            # teacher network loss
            img_loss0 = img2mse(extras['rgb0'], target_s)
            loss0 = img_loss0
            psnr0 = mse2psnr(img_loss0)

            # student network loss
            img_loss = img2mse(rgb, target_s)
            loss = img_loss
            psnr = mse2psnr(img_loss)

            # distillation loss
            loss_dist = img2mse(extras['rgb0'], rgb)
            loss += img_loss0 + loss_dist      

            # ssim0 = 1 - tf.reduce_mean(tf.image.ssim(target_s, extras['rgb0'], 1.0))  
            # ssim = 1 - tf.reduce_mean(tf.image.ssim(target_s, rgb, 1.0))     

        gradients = tape.gradient(loss0, grad_vars_teacher)
        optimizer.apply_gradients(zip(gradients, grad_vars_teacher))
        gradients = tape.gradient(loss, grad_vars_student)
        optimizer.apply_gradients(zip(gradients, grad_vars_student))

        dt = time.time()-time0
        iter_time.append(dt)

        #####           end            #####

        # Rest is logging
        # 17
        def save_weights(net, prefix, i):
            path = os.path.join(
                basedir, expname, '{}_{:06d}.npy'.format(prefix, i))
            np.save(path, net.get_weights())
            print('saved weights at', path)

        if i % args.i_weights == 0: # i_weights로 나눠떨어지면 weights를 save
            for k in models:
                print(models[k], k, i, 'asdf')
                save_weights(models[k], k, i)

        if i % args.i_video == 0 and i > 0: # i_video로 나눠떨어지면 video 생성
            rgbs, disps = render_path(
                render_poses, hwf, args.chunk, render_kwargs_test)
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(
                basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            imageio.mimwrite(moviebase + 'rgb.mp4',
                             to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4',
                             to8b(disps / np.max(disps)), fps=30, quality=8)

            if args.use_viewdirs:
                render_kwargs_test['c2w_staticcam'] = render_poses[0][:3, :4]
                rgbs_still, _ = render_path(
                    render_poses, hwf, args.chunk, render_kwargs_test)
                render_kwargs_test['c2w_staticcam'] = None
                imageio.mimwrite(moviebase + 'rgb_still.mp4',
                                 to8b(rgbs_still), fps=30, quality=8)
                
        if i % args.i_testset == 0 and i > 0: # i_testset으로 나눠떨어진다면 0, 8, 16 index에 해당하는 이미지의 pose를 통해 새로운 이미지를 prediction하고 저장. 
            testsavedir = os.path.join(
                basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses[i_test].shape) # (3, 3, 4) test set 세장에 대해서
            render_path(poses[i_test], hwf, args.chunk, render_kwargs_test,
                        gt_imgs=images[i_test], savedir=testsavedir)
            print('Saved test set')

        if i % args.i_print == 0 or i < 10: # train 시 중간과정을 print

            print(expname, i, loss_dist.numpy(), psnr0.numpy(), loss0.numpy(), psnr.numpy(), loss.numpy(), global_step.numpy())
            print('iter time {:.05f}'.format(dt))
            with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_print):
                tf.contrib.summary.scalar('loss_dist', loss_dist)
                tf.contrib.summary.scalar('loss', loss)
                tf.contrib.summary.scalar('loss0', loss0)
                tf.contrib.summary.scalar('psnr', psnr)
                tf.contrib.summary.histogram('tran', trans)
                tf.contrib.summary.scalar('psnr0', psnr0)
                # tf.contrib.summary.scalar('ssim0', ssim0)
                # tf.contrib.summary.scalar('ssim', ssim)

            if i % args.i_img == 0: # i_img로 나눠떨어질때마다 val_img들을 생성
                # Log a rendered validation view to Tensorboard
                img_i = np.random.choice(i_val) # 0, 8, 16 index인 val 이미지들 중 하나를 랜덤하게 선택
                target = images[img_i]
                pose = poses[img_i, :3, :4]

                rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, c2w=pose,
                                                **render_kwargs_test) # pose를 통해 이미지를 prediction하고 
                
                val_loss0 = img2mse(extras['rgb0'], target) # teacher network
                val_psnr0 = mse2psnr(val_loss0)
                val_loss = img2mse(rgb, target) # student network
                val_psnr = mse2psnr(val_loss) # prediction한 이미지와 원래 이미지와의 psnr 계산
                print("val_loss0 :", val_loss0.numpy(), "val_psnr0 :", val_psnr0.numpy(), "val_loss :", val_loss.numpy(), "val_psnr :", val_psnr.numpy())
                val_loss_psnr.append((i, val_loss0.numpy(), val_psnr0.numpy(), val_loss.numpy(), val_psnr.numpy()))
                
                # Save out the validation image for Tensorboard-free monitoring
                testimgdir = os.path.join(basedir, expname, 'tboard_val_imgs')
                try:
                    os.makedirs(testimgdir, exist_ok=True)
                except:
                    continue                
                # if i==0:
                #     os.makedirs(testimgdir, exist_ok=True)
                imageio.imwrite(os.path.join(testimgdir, '{:06d}.png'.format(i)), to8b(rgb))

                with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):

                    tf.contrib.summary.image('rgb', to8b(rgb)[tf.newaxis])
                    tf.contrib.summary.image(
                        'disp', disp[tf.newaxis, ..., tf.newaxis])
                    tf.contrib.summary.image(
                        'acc', acc[tf.newaxis, ..., tf.newaxis])

                    tf.contrib.summary.scalar('psnr_holdout0', val_psnr0)
                    tf.contrib.summary.scalar('psnr_holdout', val_psnr)
                    tf.contrib.summary.image('rgb_holdout', target[tf.newaxis])

                if args.N_importance > 0:

                    with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):
                        tf.contrib.summary.image(
                            'rgb0', to8b(extras['rgb0'])[tf.newaxis])
                        tf.contrib.summary.image(
                            'disp0', extras['disp0'][tf.newaxis, ..., tf.newaxis])
                        # tf.contrib.summary.image(
                        #     'z_std', extras['z_std'][tf.newaxis, ..., tf.newaxis])

        global_step.assign_add(1)


def time_calc(start, end, iter):
    t = int(end - start)
    sum = 0
    for i in iter:
        sum += i
    data = str(int(t / 3600)) + "시간 " + str(int(t % 3600 / 60)) + "분 " + str(t % 3600 % 60) + "초" + " | " + str(round(sum/len(iter), 5)) + "초"
    return data

if __name__ == '__main__':
    start = time.time()
    start_data = time.strftime('%Y-%m-%d %I:%M:%S %p', time.localtime(start))
    train()
    end = time.time()
    end_data = time.strftime('%Y-%m-%d %I:%M:%S %p', time.localtime(end))
    time_data = start_data + " | " + end_data + " | " + time_calc(start, end, iter_time) + "\n"
    print(time_data)

    f = open("logs/fortress_test/logs.txt", 'w')

    f.write(time_data)
    for i in val_loss_psnr:
        data = str(i[0]) + " " + str(i[1]) + " " + str(i[2])+ " " + str(i[3]) + " " + str(i[4]) + "\n"
        f.write(data)
    f.close()
