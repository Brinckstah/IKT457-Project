import subprocess


if __name__ == "__main__":
    from ctypes import *
    so_file = "/Users/tbmor/ikt457/IKT457-Project\my_functions.so"
    my_functions = CDLL(so_file)
    print(my_functions.main())
    