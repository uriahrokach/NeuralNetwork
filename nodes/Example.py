import nodes.Node as Node


def main():
    data = ((3, 1.5, 1),
            (2, 1, 0),
            (4, 1.5, 1),
            (3, 1, 0),
            (3.5, 0.5, 1),
            (2, 0.5, 0),
            (5.5, 1, 1),
            (1, 1, 0))

    neuron = Node.Node(data_list=data)
    neuron.train(lrn_rate=0.2, iterations=50000)
    print(neuron.weights)
    print(neuron.bias)
    print(neuron.data_list)


main()



