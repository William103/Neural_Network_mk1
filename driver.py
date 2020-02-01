from network import *
import manager

def main():
    inputs = [[0,0],[1,0],[0,1],[1,1]]
    outputs = [[0],[1],[1],[0]]
    final = manager.train_nets(inputs, outputs, 0.1, 20, 4, 0.1, 100, [2, 5, 5, 1],
            [sigmoid] * 3, [d_sigmoid] * 3, squared_error, d_squared_error, 200)
    final.train(inputs, outputs, 0.1, 3000, 4, True)

if __name__ == '__main__':
    main()
