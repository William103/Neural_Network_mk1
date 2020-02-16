from network import *
from main_thread import *

def main():
    inputs = [[0,0],[1,0],[0,1],[1,1]]
    outputs = [[0],[1],[1],[0]]
    architecture = [2, 5, 5, 1]
    f_activations = [sigmoid] * 3
    d_f_activations = [d_sigmoid] * 3
    f_cost = squared_error
    d_f_cost = d_squared_error
    random_limit = 30
    num_threads = 4
    training_rate = 10
    batch_size = 4
    epochs = 500
    master_thread = main_thread(architecture, f_activations, d_f_activations, f_cost, d_f_cost, random_limit, num_threads, inputs, outputs, training_rate, batch_size, epochs)
    master_thread.start()
    master_thread.join()
    for i in range(len(inputs)):
        print(master_thread.networks[0].prop_to_and_fro(inputs[i], outputs[i], 1), outputs[i])

if __name__ == '__main__':
    main()
