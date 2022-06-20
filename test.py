from absl import app, flags, logging
from absl.flags import FLAGS
from glob import glob
import cv2
import os
import numpy as np
import tensorflow as tf

from modules.evaluations import get_val_data, perform_val
from modules.models import ArcFaceModel
from modules.utils import set_memory_growth, load_yaml, l2_norm


flags.DEFINE_string('cfg_path', './configs/arc_res50.yaml', 'config file path')
flags.DEFINE_string('gpu', '0', 'which gpu to use')
flags.DEFINE_string('img_path', '', 'path to input image')


def main(_argv):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    logger = tf.get_logger()
    logger.disabled = True
    logger.setLevel(logging.FATAL)
    set_memory_growth()

    cfg = load_yaml(FLAGS.cfg_path)

    model = ArcFaceModel(size=cfg['input_size'],
                         backbone_type=cfg['backbone_type'],
                         training=False)

    model2 = ArcFaceModel(size=cfg['input_size'],
                         backbone_type=cfg['backbone_type'],
                         num_classes=300,
                         training=True)

    ckpt_path = tf.train.latest_checkpoint('./checkpoints/' + cfg['sub_name'])
    if ckpt_path is not None:
        print("[*] load ckpt from {}".format(ckpt_path))
        model.load_weights(ckpt_path)
    else:
        print("[*] Cannot find ckpt from {}.".format(ckpt_path))
        exit()

    if FLAGS.img_path:
        classes = {k: v for k, v in enumerate([z.split('/')[-1] for z in glob(f"{FLAGS.img_path}/*", recursive=False)])}
        imgs = glob(f"{FLAGS.img_path}/**/*.*", recursive=True)

        print(model.summary())
        for img_fn in imgs:
            img = cv2.imread(img_fn)
            img = cv2.resize(img, (224, 224))
            img = img.astype(np.float32) / 255.
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if len(img.shape) == 3:
                img = np.expand_dims(img, 0)
            embd = model(img)

            a = model2.predict([img, embd])
            print(classes[np.argmax(a)], img_fn)
        # print("[*] Encode {} to ./output_embeds.npy".format(FLAGS.img_path))
        # img = cv2.imread(FLAGS.img_path)
        # img = cv2.resize(img, (cfg['input_size'], cfg['input_size']))
        # img = img.astype(np.float32) / 255.
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # if len(img.shape) == 3:
        #     img = np.expand_dims(img, 0)
        # embeds = l2_norm(model(img))
        # np.save('./output_embeds.npy', embeds)
    else:
        print("[*] Loading LFW, AgeDB30 and CFP-FP...")
        lfw, agedb_30, cfp_fp, lfw_issame, agedb_30_issame, cfp_fp_issame = \
            get_val_data(cfg['test_dataset'])

        print("[*] Perform Evaluation on LFW...")
        acc_lfw, best_th = perform_val(
            cfg['embd_shape'], cfg['batch_size'], model, lfw, lfw_issame,
            is_ccrop=cfg['is_ccrop'])
        print("    acc {:.4f}, th: {:.2f}".format(acc_lfw, best_th))

        # print("[*] Perform Evaluation on AgeDB30...")
        # acc_agedb30, best_th = perform_val(
        #     cfg['embd_shape'], cfg['batch_size'], model, agedb_30,
        #     agedb_30_issame, is_ccrop=cfg['is_ccrop'])
        # print("    acc {:.4f}, th: {:.2f}".format(acc_agedb30, best_th))

        # print("[*] Perform Evaluation on CFP-FP...")
        # acc_cfp_fp, best_th = perform_val(
        #     cfg['embd_shape'], cfg['batch_size'], model, cfp_fp, cfp_fp_issame,
        #     is_ccrop=cfg['is_ccrop'])
        # print("    acc {:.4f}, th: {:.2f}".format(acc_cfp_fp, best_th))


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
