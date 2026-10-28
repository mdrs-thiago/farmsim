import sys
from rqt_gui.main import Main


def main():
    main_ = Main()
    sys.exit(main_.main(sys.argv, standalone='thorvald_gui/ThorvaldGUI'))


if __name__ == '__main__':
    main()
