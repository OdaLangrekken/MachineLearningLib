import numpy 

class LogisticRegression:
        
        def __init__(self, num_classes = 2):
            self.num_classes = num_classes
                            
        def sigmoid(self, z):
            sig = 1 / (1 + numpy.exp(-z))
            return sig

        def cost(self, theta, x, y, reg_type, lamb):
            m = len(y)
            sigmoids = self.sigmoid(numpy.dot(x, theta))
            first_log = numpy.log(sigmoids)
            second_log = numpy.log(1 - sigmoids)
            test = -y*first_log - (1 - y)*second_log
            if reg_type == 'L2': 
                J = sum(test)/m + lamb/(2*m) * sum(theta[1:]**2)
            elif reg_type == 'L1':
                J = sum(test)/m + lamb/(2*m) * sum(abs(theta[1:]))
            elif reg_type == 'elastic':
                J = sum(test)/m + lamb/(2*m) * sum(abs(theta[1:])) + + lamb/(2*m) * sum(theta[1:]**2)
                                                                                                                                 
            grad = 1/m*numpy.dot(x.T,(sigmoids - y))
            grad[1:] = grad[1:] + lamb/m * theta[1:]
            return J, grad

        def gradient_descent(self, theta, x, y, reg_type, lamb, alpha, max_iters):
            it = 0
            while it < max_iters:
                J, grad = self.cost(theta, x, y, reg_type, lamb)
                theta = theta - alpha * grad  
                it += 1
            return theta

        def fit(self, X_train, Y_train, reg_type = 'L2', lamb = 0, alpha = 0.1, max_iters = 100):
            import numpy, time
            m = X_train.shape[0]
            n = X_train.shape[1]

            self.theta = numpy.ndarray(shape=(self.num_classes, n + 1), dtype=float) 
            initial_theta = numpy.zeros(n+1)

            X_train_fix = numpy.concatenate((numpy.ones((m, 1)), X_train), axis=1)
                                                             
            start = time.time()
            for c in range(self.num_classes):
                print(f'Trainings done: {c}/{self.num_classes}')
                y_c = numpy.array([int(y_val == c) for y_val in Y_train])
                theta_new = self.gradient_descent(initial_theta, X_train_fix, y_c, reg_type, lamb, alpha, max_iters)
                theta_new = theta_new.reshape(theta_new.shape[0], 1)
                self.theta[c, :] = numpy.transpose(theta_new)
                end = time.time()
                if (end - start)/60 < 1:
                    print(f'Time elapsed: {end - start:.2f} s')
                else:
                    print(f'Time elapsed: {(end - start)/60:.2f} m')

        def predict(self, X_test):
            m2 = X_test.shape[0]
            X_test_fix = numpy.concatenate((numpy.ones((m2, 1)), X_test), axis=1)
            hyp = self.theta.dot(numpy.transpose(X_test_fix))
            hyp = numpy.transpose(hyp)

            y_pred = []
            for i in range(len(hyp)):
                data = list(hyp[i, :])
                most_likely = data.index(max(data))
                y_pred.append(most_likely)
            return(y_pred)

        def accuracy(self, y_test, y_pred):
            acc = 0
            for i in range(len(y_pred)):
                if y_pred[i] == y_test[i]:
                    acc += 1
            return acc/len(y_pred)
