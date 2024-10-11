import os
import warnings
from absl import app, flags

import torch
from torchvision.utils import make_grid, save_image
from tqdm import trange
import torch.distributed as dist

from diffusion import GaussianDiffusionSampler
from model import UNet
from score.both import get_inception_and_fid_score

device = torch.device('cuda:0')

FLAGS = flags.FLAGS
# UNet
flags.DEFINE_integer('ch', 128, help='base channel of UNet')
flags.DEFINE_multi_integer('ch_mult', [1, 2, 2, 2], help='channel multiplier')
flags.DEFINE_multi_integer('attn', [1], help='add attention to these levels')
flags.DEFINE_integer('num_res_blocks', 2, help='# resblock in each level')
flags.DEFINE_float('dropout', 0., help='dropout rate of resblock')
# Gaussian Diffusion
flags.DEFINE_enum('mean_type', 'xstart', ['xstart', 'epsilon'], help='predict variable')
flags.DEFINE_enum('var_type', 'fixedlarge', ['fixedlarge', 'fixedsmall'], help='variance type')
# Training
flags.DEFINE_integer('img_size', 32, help='image size')
flags.DEFINE_integer('batch_size', 128, help='batch size')
flags.DEFINE_integer('num_workers', 4, help='workers of Dataloader')
flags.DEFINE_string('gpu_id', '0', help='single gpu training')
flags.DEFINE_bool('conditional', False, help='use conditional or not')
flags.DEFINE_integer('class_num', 10, help='class num')
# Logging & Sampling
flags.DEFINE_string('logdir', './logs/CIFAR10/1024', help='log directory')
flags.DEFINE_integer('stride', 1, help='sampling stride')
flags.DEFINE_integer('num_images', 50000, help='the number of generated images for evaluation')
flags.DEFINE_bool('fid_use_torch', False, help='calculate IS and FID on gpu')
flags.DEFINE_string('fid_cache', './stats/cifar10.train.npz', help='FID cache')
flags.DEFINE_integer('seed', 0, help='seed')

def evaluate(sampler, model):
    model.eval()
    with torch.no_grad():
        images = []
        desc = "generating images"
        for i in trange(0, FLAGS.num_images, FLAGS.batch_size, desc=desc):
            batch_size = min(FLAGS.batch_size, FLAGS.num_images - i)
            x_T = torch.randn((batch_size, 3, FLAGS.img_size, FLAGS.img_size))
            y_target = torch.randint(FLAGS.class_num, size=(x_T.shape[0],), device=device)
            batch_images = sampler.ddim(x_T.to(device), FLAGS.stride, y=y_target).cpu()
            images.append((batch_images + 1) / 2)
        images = torch.cat(images, dim=0).numpy()
    model.train()
    (IS, IS_std), FID = get_inception_and_fid_score(
        images, FLAGS.fid_cache, num_images=FLAGS.num_images,
        use_torch=FLAGS.fid_use_torch, verbose=True)
    return (IS, IS_std), FID, images


def eval():
    ckpt = torch.load(os.path.join(FLAGS.logdir, 'ckpt.pt'), map_location='cuda:0')
    T = ckpt['T']
    time_scale = ckpt['time_scale']
    # model setup
    model = UNet(
        T=T*time_scale, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout,
        conditional=FLAGS.conditional, class_num=FLAGS.class_num)
    sampler = GaussianDiffusionSampler(
        model, T, time_scale, img_size=FLAGS.img_size,
        mean_type=FLAGS.mean_type, var_type=FLAGS.var_type).to(device)

    # load model and evaluate
    if not os.path.exists(os.path.join(FLAGS.logdir, 'ddim_stride' + str(FLAGS.stride))):
        os.makedirs(os.path.join(FLAGS.logdir, 'ddim_stride' + str(FLAGS.stride)))
    if time_scale != 1:
        model.load_state_dict(ckpt['net_model'])
        del ckpt
        (IS, IS_std), FID, samples = evaluate(sampler, model)
        print("Model     : IS:%6.3f(%.3f), FID:%7.3f" % (IS, IS_std, FID))
        with open(os.path.join(FLAGS.logdir, 'ddim_stride' + str(FLAGS.stride), 'result_seed' + str(FLAGS.seed) + '.txt'), 'w') as f:
            f.write('IS: ' + str(IS))
            f.write('IS_std: ' + str(IS_std))
            f.write('FID: ' + str(FID))
        save_image(
            torch.tensor(samples[:256]),
            os.path.join(FLAGS.logdir, 'ddim_stride' + str(FLAGS.stride), 'samples_seed' + str(FLAGS.seed) + '.png'),
            nrow=16)
    else:
        model.load_state_dict(ckpt['ema_model'])
        del ckpt
        (IS, IS_std), FID, samples = evaluate(sampler, model)
        print("Model(EMA): IS:%6.3f(%.3f), FID:%7.3f" % (IS, IS_std, FID))
        with open(os.path.join(FLAGS.logdir, 'ddim_stride' + str(FLAGS.stride), 'result_seed' + str(FLAGS.seed) + '.txt'), 'w') as f:
            f.write('IS: ' + str(IS))
            f.write('IS_std: ' + str(IS_std))
            f.write('FID: ' + str(FID))
        save_image(
            torch.tensor(samples[:256]),
            os.path.join(FLAGS.logdir, 'ddim_stride' + str(FLAGS.stride), 'samples_seed' + str(FLAGS.seed) + '.png'),
            nrow=16)


def main(argv):
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_id
    seed = FLAGS.seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    # Fully deterministic
    # suppress annoying inception_v3 initialization warning
    warnings.simplefilter(action='ignore', category=FutureWarning)
    eval()


if __name__ == '__main__':
    app.run(main)
