from network import *

def main():
    test = FeedForwardNetwork([2, 3, 3, 1], [itself, itself, itself],
            [d_itself, d_itself, d_itself], squared_error, d_squared_error)
    inputs = [[0,0],[1,0],[0,1],[1,1]]
    outputs = [[0],[1],[1],[0]]
    for inpt in inputs:
        print(test.prop(inpt))
    input()
    for i in range(1):
        test.train(inputs, outputs, 0.1, 2, 4)
        for inpt in inputs:
            print(test.prop(inpt))

if __name__ == '__main__':
    main()
