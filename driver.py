from network import *

def main():
    test = FeedForwardNetwork([2, 3, 2], [sigmoid, relu], [d_sigmoid, d_relu])
    print(test.prop([1, 2]))

if __name__ == '__main__':
    main()