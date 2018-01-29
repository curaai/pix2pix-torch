import argparse

from model import Pix2Pix

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=15, type=int)
    parser.add_argument('--epoch_iter', default=1000, type=int)
    parser.add_argument('--lr', default=0.00001, type=float)
    parser.add_argument('--src_path', default='data/source/crop_gray', type=str, help="src iamge dataset")
    parser.add_argument('--trg_path', default='data/target/crop_color', type=str, help="trg image dataset")
    parser.add_argument('--sample_path', default='samples', type=str, help="save sample images while training")
    parser.add_argument('--save_model_path', default='save', type=str, help="save trained model")
    parser.add_argument('--restore_model_path', default='save', type=str, help="restore trained model")
    parser.add_argument('--use_gpu', default=True, type=bool)
    args = parser.parse_args()

    model = Pix2Pix(args.batch_size, args.epoch_iter, args.lr, args.src_path, args.trg_path, args.sample_path, args.save_model_path, args.restore_model_path, args.use_gpu)
    model.train()