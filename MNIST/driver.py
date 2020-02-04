from network import *
import mnist
import manager

def main():
    training_data, validation_data, test_data = mnist.load_data_wrapper()
    #training_inputs, training_outputs = zip(*training_data)
    #validation_inputs, validation_outputs = zip(*validation_data)
    #test_inputs, test_outputs = zip(*test_data)
    training_data = list(training_data)
    validation_data = list(validation_data)
    test_data = list(test_data)

    batch_size = 100
    training_rate = 0.1
    min_error = 0.001
    net = FeedForwardNetwork([784, 50, 50, 10], [sigmoid] * 3, [d_sigmoid] * 3, squared_error, d_squared_error, 10)
    net.train_to_accuracy(training_data, validation_data, training_rate, min_error, batch_size, True)
    correct = 0
    total_error = 0
    maxdex = 0
    maximum = -1
    print('Validating network')
    for data_point in validation_data]:
        output, error = net.prop_to_and_fro(data_point[0], data_point[1], 0)
        total_error += error
        if np.argmin(output) == np.argmin(data_point[1]):
            correct += 1

    print("Done! Final accuracy: " + str(correct / len(validation_data)) + '%')

if __name__ == '__main__':
    main()
