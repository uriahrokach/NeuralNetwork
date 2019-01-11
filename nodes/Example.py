import nodes.Node as Node


def main():
    # This is a data that represents a flower.
    data = ((3, 1.5, 1),
            (2, 1, 0),
            (4, 1.5, 1),
            (3, 1, 0),
            (3.5, 0.5, 1),
            (2, 0.5, 0),
            (5.5, 1, 1),
            (1, 1, 0))

    neuron = Node.Node(data_list=data)
    neuron.train(lrn_rate=0.2, iterations=1000000)
    while True:
        result = int(100 * neuron.run_node())
        if result > 50:
            print("the node is sure in "+str(result)+"% that this is a red flower.")
        else:
            print("the node is sure in " + str(100 - result) + "% that this is a blue flower.")


main()



