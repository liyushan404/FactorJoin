import argparse
import pickle


def get_pfl(pkl_path):
    with open(pkl_path, 'rb') as f:
        a = pickle.load(f)
        print(a[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl_path', default='/home/lrr/Documents/FactorJoin/checkpoints/derived_query_file.pkl',
                        help='Which dataset to be used')
    args = parser.parse_args()
    get_pfl(args.pkl_path)