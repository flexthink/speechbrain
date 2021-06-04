from argparse import ArgumentParser
from hyperpyyaml import load_hyperpyyaml
import os

def main():
    parser = ArgumentParser()
    parser.add_argument(
        '--model',
        help='The model to load')
    parser.add_argument(
        '--file',
        help='The YAML file to load')
    arguments = parser.parse_args()
    if arguments.file:
        hparams_path = arguments.file
    elif arguments.model:
        hparams_path = os.path.join(arguments.model, 'hyperparams.yaml')
    else:
        print("Either --model or --file needs to be specified")
        sys.exit(1)
    with open(hparams_path) as hparams_file:
        hparams = load_hyperpyyaml(hparams_file)
    print(f"Loaded: {hparams_path}")
    for key in hparams.keys():
        print(key)

if __name__ == '__main__':
    main()
