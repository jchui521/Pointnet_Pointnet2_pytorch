"""
Author: Benny
"""
Wrapper kept for backward compatibility.
Delegates to examples.original.classification.train
"""

from examples.original.classification.train import main, parse_args


if __name__ == '__main__':
    args = parse_args()
    main(args)
import importlib
