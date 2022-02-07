"""Main script."""

import argparse
from agents import ALGORITHMS

def train(args):
    """Train algorithm.
    Args:
        args (argparse.Namespace): argparse arguments.
    """
    algorithm = ALGORITHMS[args.algorithm]
    algorithm.create_algorithm()
    algorithm.train()

def test(args):
    """Test algorithm.
    Args:
        args (argparse.Namespace): argparse arguments.
    """
    algorithm = ALGORITHMS[args.algorithm]
    algorithm.create_algorithm()
    algorithm.test()

if __name__ == "__main__":

    # Parser
    parser = argparse.ArgumentParser(description="Main script")
    # Subparsers
    subparsers = parser.add_subparsers()

    # Algorithm name
    parser.add_argument("--algorithm",
                        choices=list(ALGORITHMS.keys()),
                        help="Algorithm name.",
                        required=True,
                        type=str)

    # Subparser
    train_parser = subparsers.add_parser("train", help="Train algorithm")
    train_parser.set_defaults(func=train)

    test_parser = subparsers.add_parser("test", help="Test algorithm")
    test_parser.set_defaults(func=test)

    # Parse arguments
    args = parser.parse_args()
    args.func(args)
