import os
import sys
from toolkit_dslr.graph_utils import draw_scatter_plot 


def main():
    try:
        assert len(sys.argv) > 1, "File path is missing"
        assert os.path.exists(sys.argv[1]), "The file does not exists"

        draw_scatter_plot(sys.argv[1])

    except AssertionError as error:
        print(AssertionError.__name__ + ":", error)


if __name__ == "__main__":
    main()
