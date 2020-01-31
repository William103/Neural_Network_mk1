from network import *

def main():
    test = FeedForwardNetwork([2, 3, 3, 1], [sigmoid, sigmoid, sigmoid],
            [d_sigmoid, d_sigmoid, d_sigmoid], squared_error, d_squared_error)
    inputs = [[0,0],[1,0],[0,1],[1,1]]
    outputs = [[0],[1],[1],[0]]
    input()
    for i in range(1):
        test.train(inputs, outputs, 0.1, 100, 4)

if __name__ == '__main__':
    main()
