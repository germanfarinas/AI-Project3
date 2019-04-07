import numpy as np
import pandas as pd

# Will use position as an object to hold the weights of the neural network


class Position:
    def __init__(self, input_to_hidden_weights, hidden_to_output_weights):
        self.input_to_hidden = input_to_hidden_weights
        self.hidden_to_output = hidden_to_output_weights

    def get_input_to_hidden_weights(self):
        return self.input_to_hidden

    def set_input_to_hidden_weights(self, input_to_hidden):
        self.input_to_hidden = input_to_hidden

    def get_hidden_to_output_weights(self):
        return self.hidden_to_output

    def set_hidden_to_output_weights(self, hidden_to_output):
        self.hidden_to_output = hidden_to_output


class Velocity:
    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        self.input_to_hidden_velocity = np.zeros((input_nodes, hidden_nodes))
        self.hidden_to_output_velocity = np.zeros((hidden_nodes, output_nodes))

    def get_input_to_hidden_velocities(self):
        return self.input_to_hidden_velocity

    def set_input_to_hidden_velocites(self, new_velocities):
        self.input_to_hidden_velocity = new_velocities

    def get_hidden_to_output_velocities(self):
        return self.hidden_to_output_velocity

    def set_hidden_to_output_velocities(self, new_velocities):
        self.hidden_to_output_velocity = new_velocities


class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        self.input_layer_nodes = input_nodes
        self.hidden_layer_nodes = hidden_nodes
        self.output_layer_nodes = output_nodes

        # Randomly initialize theta (MLP weights)
        #self.input_to_hidden_weights, self.hidden_to_output_weights = self.initialize_weights()

        w1, w2 = self.initialize_weights()
        self.position = Position(w1, w2)

    def get_num_input_layer_nodes(self):
        return self.input_layer_nodes

    def get_num_hidden_layer_nodes(self):
        return self.hidden_layer_nodes

    def get_num_output_layer_nodes(self):
        return self.output_layer_nodes

    def get_position(self):
        return self.position

    def set_position(self, position):
        self.position = position

    def initialize_weights(self):
        w1 = np.random.random((self.input_layer_nodes, self.hidden_layer_nodes)) * 0.2 - 0.1
        w2 = np.random.random((self.hidden_layer_nodes, self.output_layer_nodes)) * 0.2 - 0.1
        return w1, w2

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def feedforward(self, X):
        """
        :param X: input vector
        :return: output vector after X has been fed through network
        """

        # input to hidden layer activation function
        input_hidden = X.dot(self.position.get_input_to_hidden_weights())
        output_hidden = self.sigmoid(input_hidden)

        input_output = output_hidden.dot(self.position.get_hidden_to_output_weights())
        output = self.sigmoid(input_output)

        return output

    def calculate_error(self, X, Y):
        """

        :param X: input vector
        :param Y: target vector
        :return: MSE
        """

        # return (np.square(Y - self.feedforward(X)))/Y.shape[0]
        return np.square(Y-self.feedforward(X)).mean(axis=None)

class Particle:
    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        self.network = NeuralNetwork(input_nodes, hidden_nodes, output_nodes)
        self.position = self.network.get_position()

        self.current_cost = None
        # first position is initially best
        self.best_position = self.position
        self.best_cost = 999999999

        self.velocity = Velocity(input_nodes, hidden_nodes, output_nodes)

    def get_cost(self, X, Y):
        """

        :param X: input vector
        :param Y: target vector
        :return:
        """
        return self.network.calculate_error(X, Y)

    def get_position(self):
        return self.position

    def get_best_position(self):
        return self.best_position

    def get_velocity(self):
        return self.velocity

    def get_best_position(self):
        return self.best_position

    def get_best_cost(self):
        return self.best_cost

    def set_best_cost(self, cost):
        self.best_cost = cost

    def set_velocity(self, input_to_hidden, hidden_to_output):
        self.velocity.set_input_to_hidden_velocites(input_to_hidden)
        self.velocity.set_hidden_to_output_velocities(hidden_to_output)

    def set_position(self, input_to_hidden, hidden_to_output):
        self.position.set_input_to_hidden_weights(input_to_hidden)
        self.position.set_hidden_to_output_weights(hidden_to_output)
        self.network.set_position(self.position)

    def set_best_position(self, position):
        self.best_position = position


class ParticleSwarm:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, num_particles):
        self.particles = list()
        for i in range(num_particles):
            self.particles.append(Particle(input_nodes, hidden_nodes, output_nodes))

        self.swarm_best_position = None
        self.swarm_best_cost = None

    def get_swarm_best_position(self):
        return self.swarm_best_position

    def set_swarm_best_position(self, position):
        self.swarm_best_position = position

    def get_swarm_best_cost(self):
        return self.swarm_best_cost

    def set_swarm_best_cost(self, cost):
        self.swarm_best_cost = cost

    def get_initial_bests(self, X, Y):
        for particle in self.particles:
            # to set best to only current known
            if self.swarm_best_position is None:
                self.set_swarm_best_position(particle.get_position())
            if self.swarm_best_cost is None:
                self.set_swarm_best_cost(particle.get_cost(X, Y))

            # find best cost out of initial weights
            particle_cost = particle.get_cost(X, Y)
            if particle_cost < self.swarm_best_cost:
                self.set_swarm_best_cost(particle_cost)
                particle_position = particle.get_position()
                self.set_swarm_best_position(particle_position)

    def update_velocity(self, particle, inertia_weight, local_learn, global_learn):
        current_input_to_hidden = particle.get_velocity().get_input_to_hidden_velocities()
        current_hidden_to_output = particle.get_velocity().get_hidden_to_output_velocities()

        rows = current_input_to_hidden.shape[0]
        columns = current_input_to_hidden.shape[1]

        updated_input_to_hidden = inertia_weight * current_input_to_hidden + local_learn * np.multiply((np.random.rand(rows, columns)),  particle.get_best_position().get_input_to_hidden_weights()) + global_learn * np.multiply(np.random.rand(rows, columns), self.get_swarm_best_position().get_input_to_hidden_weights())

        rows = current_hidden_to_output.shape[0]
        columns = current_hidden_to_output.shape[1]
        updated_hidden_to_output = inertia_weight * current_hidden_to_output + local_learn * np.multiply(np.random.rand(rows, columns), particle.get_best_position().get_hidden_to_output_weights()) + global_learn * np.multiply(np.random.rand(rows, columns), self.get_swarm_best_position().get_hidden_to_output_weights())

        particle.set_velocity(updated_input_to_hidden, updated_hidden_to_output)

    def train(self, X, Y, iterations, inertia_weight, local_learn, global_learn, inertia_damping):

        #find min_best_cost
        self.get_initial_bests(X, Y)

        for i in range(iterations):
            for particle in self.particles:
                self.update_velocity(particle, inertia_weight, local_learn, global_learn)

                # Velocity limits??

                particle.set_position(particle.get_position().get_input_to_hidden_weights() + particle.get_velocity().get_input_to_hidden_velocities(),
                                      particle.get_position().get_hidden_to_output_weights() + particle.get_velocity().get_hidden_to_output_velocities())

                # Velocity mirror effect?

                # position limits

                error = particle.get_cost(X, Y)

                if error < particle.get_best_cost():
                    particle.set_best_position(particle.get_position())
                    particle.set_best_cost(error)

                    if error < self.get_swarm_best_cost():
                        self.set_swarm_best_cost(error)
                        self.set_swarm_best_position(particle.get_position())

            inertia_weight = inertia_weight * inertia_damping
            print("Best error is")
            print(self.get_swarm_best_cost())


# file="Documents\\University_of_Miami\\Machine_Learning\\ionosphere.txt"
file = "C:\\Users\\Dell\\Documents\\University_of_Miami\\Fall 2018\\Machine_Learning\\ionosphere.txt"
data = pd.read_csv(file,delim_whitespace=False,header=-1)


data = np.array(data)
np.random.shuffle(data)
training_data = data[0:int(len(data)*.8)]
testing_data = data[int(len(data)*.8):len(data)]

# Training data
train_X = training_data[:, 0:34]
print(train_X[0])
train_X = train_X.astype(float)
train_y = training_data[:, 34:35]
print(train_y[0])
print('Shape of training set: ' + str(train_X.shape))
print('Shape of training set class: ' + str(train_y.shape))

# change y [1D] to Y [2D] sparse array coding class
n_examples = len(train_y)
labels = np.unique(train_y)
train_Y = np.zeros((n_examples, len(labels)))
for ix_label in range(len(labels)):
    # Find examples with with a Label = lables(ix_label)
    ix_tmp = np.where(train_y == labels[ix_label])[0]
    train_Y[ix_tmp, ix_label] = 1
print(labels)
# print(train_Y)

# Test data
test_X = testing_data[:, 0:34]
test_X = test_X.astype(float)
test_y = testing_data[:, 34:35]
print('Shape of test set: ' + str(test_X.shape))

# change y [1D] to Y [2D] sparse array coding class
n_examples = len(test_y)
labels = np.unique(test_y)
test_Y = np.zeros((n_examples, len(labels)))
for ix_label in range(len(labels)):
    # Find examples with with a Label = lables(ix_label)
    ix_tmp = np.where(test_y == labels[ix_label])[0]
    test_Y[ix_tmp, ix_label] = 1
print(labels)

swarm = ParticleSwarm(34, 50, 2, 50)
swarm.train(train_X, train_Y, iterations=1000, inertia_weight=1, local_learn=1.5, global_learn=2.0, inertia_damping=0.99)





