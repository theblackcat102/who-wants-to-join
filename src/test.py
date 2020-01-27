from torch.utils.tensorboard import SummaryWriter


def dict2table(params):
    text = '\n\n'
    text = '|  Attribute  |     Value    |\n'+'|'+'-'*13+'|'+'-'*14+'|'
    for key, value in params.items():
        text += '\n|{:13}|{:14}|'.format(str(key), str(value))
    return text

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description='Deepset Recommendation Model')
    # dataset parameters
    parser.add_argument('--dataset', type=str, default='amazon',
                        choices=['amazon', 'dblp', 'youtube'])
    # training parameters
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--eval', type=int, default=10)
    parser.add_argument('--save', type=int, default=50)
    # model parameters
    parser.add_argument('--input-dim', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('-l','--layers', nargs='+', type=int, default=[32, 32, 32])
    parser.add_argument('--maxhop', type=int, default=2)

    args = parser.parse_args()
    print(isinstance(args, dict))
    print(isinstance(vars(args), dict))

    print(dict2table(vars(args)))
    # writer = SummaryWriter(log_dir='./logs/test')
    # writer.add_text('Text', dict2table(test), 0)
    # for idx in range(100):
    #     writer.add_scalar('data/test', idx)