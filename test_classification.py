"""
Wrapper kept for backward compatibility.
Delegates to examples.original.classification.test
"""

from examples.original.classification.test import main, parse_args


if __name__ == '__main__':
    args = parse_args()
    main(args)
