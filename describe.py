import sys
import os
from toolkit_dslr.describe_utils import describe


def main():
    try:
        assert len(sys.argv) == 2, "You must provide the dataset file path"
        assert os.path.exists(sys.argv[1]), "The file does not exists"
        print(describe(sys.argv[1]))
    except AssertionError as error:
        print(AssertionError.__name__ + ":", error)


if __name__ == "__main__":
    main()
