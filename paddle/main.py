import yaml
import argparse
from scrips import same_seeds,prep_loader,train,evaluate
from models import build_model
from utils import logger

parser = argparse.ArgumentParser(description='Training Config', add_help=True)
parser.add_argument('-c', '--config', default='conf/base.yaml', type=str, metavar='FILE',help='yaml file path')
parser.add_argument('-m','--mode',default='eval',type=str,choices=['train','eval'])
args=parser.parse_args()

def main(args):
    same_seeds(seed=2021)
    conf = yaml.load(open(args.config, 'r', encoding='utf-8'), Loader=yaml.FullLoader)
    logger.info('Prep | Loading datasets...')
    train_loader,dev_loader=prep_loader(conf)
    logger.info('Prep | Loading model...')
    model=build_model(conf)
    if args.mode=='train':
        logger.info('Train | Training...')
        train(conf,model,train_loader,dev_loader)
    elif args.mode=='eval':
        logger.info('Eval | Evaluating...')
        evaluate(model,dev_loader)
    else:
        logger.info('Mode error!')

if __name__ == '__main__':
    main(args)
