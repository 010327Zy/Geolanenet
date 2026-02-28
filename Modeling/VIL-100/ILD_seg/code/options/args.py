import argparse

def parse_args(cfg):
    root = '/train1/gzp/Geolanenet/GeoLaneNet'
    parser = argparse.ArgumentParser(description='Hello')
    parser.add_argument('--run_mode', type=str, default='train', help='run mode (train, test)')
    parser.add_argument('--pre_dir', type=str, default=f'{root}/Preprocessing/VIL-100', help='preprocessed data dir')
    parser.add_argument('--dataset_dir', default=f'{root}/VIL100', help='dataset dir')
    args = parser.parse_args()

    cfg = args_to_config(cfg, args)
    return cfg

# Example on my PC env
# --------------------------------------------------------
def parse_args(cfg):
    root = '/train1/gzp/Geolanenet/GeoLaneNet'
    parser = argparse.ArgumentParser(description='Hello')
    parser.add_argument('--run_mode', type=str, default='train', help='run mode (train, test)')
    parser.add_argument('--pre_dir', type=str, default=f'{root}/Preprocessing/VIL-100', help='preprocessed data dir')
    parser.add_argument('--dataset_dir', default=f'{root}/VIL100', help='dataset dir')
    args = parser.parse_args()

    cfg = args_to_config(cfg, args)
    return cfg
# --------------------------------------------------------

def args_to_config(cfg, args):
    if args.dataset_dir is not None:
        cfg.dir['dataset'] = args.dataset_dir
    if args.pre_dir is not None:
        cfg.dir['head_pre'] = args.pre_dir
        cfg.dir['pre0_train'] = cfg.dir['pre0_train'].replace('--preprocessed data path', args.pre_dir)
        cfg.dir['pre0_test'] = cfg.dir['pre0_test'].replace('--preprocessed data path', args.pre_dir)
        cfg.dir['pre2'] = cfg.dir['pre2'].replace('--preprocessed data path', args.pre_dir)

    return cfg