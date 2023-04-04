import sys

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainter, QBrush, QColor, QFont
from PyQt5.QtWidgets import QWidget, QApplication, QPushButton, QLineEdit, QComboBox
from matplotlib import pyplot as plt

from AIModel import Model
from mymath import sigma, sigma_derivative, tanh_derivative, relu, relu_derivative


class MyWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setGeometry(100, 100, 1400, 800)
        self.setWindowTitle("Test")
        plt.grid(True)
        plt.xlabel('Epoch number')
        plt.ylabel('Cost')

        # \/ \/ \/ control elements \/ \/ \/

        self.train_btn = QPushButton(self)
        self.train_btn.setFont(QFont("Serif", 14))
        self.train_btn.setText("TRAIN")
        self.train_btn.setGeometry(25, 25, 180, 35)
        self.train_btn.clicked.connect(self.train)
        self.train_btn.setAttribute(Qt.WA_OpaquePaintEvent)

        self.plot_cost = QPushButton(self)
        self.plot_cost.setFont(QFont("Serif", 14))
        self.plot_cost.setText("Plot cost")
        self.plot_cost.setGeometry(self.train_btn.x() + 205, 70, 180, 35)
        self.plot_cost.clicked.connect(self.plot_cost_fn)
        self.plot_cost.setAttribute(Qt.WA_OpaquePaintEvent)

        self.clr_btn = QPushButton(self)
        self.clr_btn.setStyleSheet("QPushButton{background: #ffaf00}")
        self.clr_btn.setFont(QFont("Serif", 14))
        self.clr_btn.setText("CLEAR")
        self.clr_btn.setGeometry(self.train_btn.x(), self.train_btn.y() + 45, 180, 35)
        self.clr_btn.clicked.connect(self.clear)
        self.clr_btn.setAttribute(Qt.WA_OpaquePaintEvent)

        self.lr_edit = QLineEdit(self)
        self.lr_edit.setFont(QFont("Serif", 14))
        self.lr_edit.setPlaceholderText("Learning rate")
        self.lr_edit.setGeometry(self.width() - 200, 25, 150, 35)
        self.lr_edit.editingFinished.connect(self.lr_change)
        self.lr_edit.setAttribute(Qt.WA_OpaquePaintEvent)

        self.epochs_edit = QLineEdit(self)
        self.epochs_edit.setFont(QFont("Serif", 14))
        self.epochs_edit.setPlaceholderText("Epochs num")
        self.epochs_edit.setGeometry(self.width() - 200, 75, 150, 35)
        self.epochs_edit.editingFinished.connect(self.epochs_change)
        self.epochs_edit.setAttribute(Qt.WA_OpaquePaintEvent)

        self.batch_edit = QLineEdit(self)
        self.batch_edit.setFont(QFont("Serif", 14))
        self.batch_edit.setPlaceholderText("Batch size")
        self.batch_edit.setGeometry(self.width() - 200, 125, 150, 35)
        self.batch_edit.editingFinished.connect(self.batch_change)
        self.batch_edit.setAttribute(Qt.WA_OpaquePaintEvent)

        self.sizes_edit = QLineEdit(self)
        self.sizes_edit.setFont(QFont("Serif", 14))
        self.sizes_edit.setPlaceholderText("Layers sizes")
        self.sizes_edit.setGeometry(self.width() - 400, 25, 150, 35)
        self.sizes_edit.editingFinished.connect(self.sizes_change)
        self.sizes_edit.setAttribute(Qt.WA_OpaquePaintEvent)

        self.activation_combox = QComboBox(self)
        self.activation_combox.setStyleSheet("QComboBox{background: #b3fffa}")
        self.activation_combox.setFont(QFont("Serif", 14))
        self.activation_combox.addItems(["sigmoid", "tanh", "ReLU"])
        self.activation_combox.setGeometry(self.width() - 400, 75, 150, 35)
        self.activation_combox.currentTextChanged.connect(self.activ_func_change)
        self.activation_combox.setAttribute(Qt.WA_OpaquePaintEvent)

        self.tune_learn_rata_btn = QPushButton(self)
        self.tune_learn_rata_btn.setFont(QFont("Serif", 14))
        self.tune_learn_rata_btn.setText("Tune learn-rate")
        self.tune_learn_rata_btn.setGeometry(self.train_btn.x() + 205, 25, 180, 35)
        self.tune_learn_rata_btn.clicked.connect(self.experiment)
        self.tune_learn_rata_btn.setAttribute(Qt.WA_OpaquePaintEvent)

        self.compare_activ_funcs_btn = QPushButton(self)
        self.compare_activ_funcs_btn.setStyleSheet("QPushButton{background: #b3fffa}")
        self.compare_activ_funcs_btn.setFont(QFont("Serif", 14))
        self.compare_activ_funcs_btn.setText("^ Compare ^")
        self.compare_activ_funcs_btn.setGeometry(self.width() - 400, 125, 150, 35)
        self.compare_activ_funcs_btn.clicked.connect(self.compare_activ_funcs)
        self.compare_activ_funcs_btn.setAttribute(Qt.WA_OpaquePaintEvent)

        self.learning_algorithm_change = QComboBox(self)
        self.learning_algorithm_change.setStyleSheet("QComboBox{background: #b7ffb3}")
        self.learning_algorithm_change.setFont(QFont("Serif", 14))
        self.learning_algorithm_change.addItems(["Batch GD", "Adam"])
        self.learning_algorithm_change.setGeometry(self.width() - 200, 190, 150, 35)
        self.learning_algorithm_change.currentTextChanged.connect(self.learn_algorithm_change)
        self.learning_algorithm_change.setAttribute(Qt.WA_OpaquePaintEvent)

        self.compare_learn_algo_btn = QPushButton(self)
        self.compare_learn_algo_btn.setStyleSheet("QPushButton{background: #b7ffb3}")
        self.compare_learn_algo_btn.setFont(QFont("Serif", 14))
        self.compare_learn_algo_btn.setText("^ Compare ^")
        self.compare_learn_algo_btn.setGeometry(self.width() - 200, 240, 150, 35)
        self.compare_learn_algo_btn.clicked.connect(self.compare_learn_algo)
        self.compare_learn_algo_btn.setAttribute(Qt.WA_OpaquePaintEvent)

        # ^^^ control elements ^^^

        # \/ \/ \/ different parameters \/ \/ \/

        self.pointsArray = np.ndarray((0, 3), dtype='float64')
        self.pointR = 10
        self.fieldInterval = 10
        self.field = np.zeros((self.width() // self.fieldInterval, self.height() // self.fieldInterval), dtype='int')

        self.sizes = [2, 5, 1]
        self.activation_function = sigma
        self.activ_func_derivative = sigma_derivative
        self.model = Model(self.sizes, self.activation_function)
        self.epochs_num = 200
        self.lr = 1
        self.batch_size = 1
        self.learn_algorithm = 'Batch GD'
        self.green_value = 1  # don't touch this
        self.red_value = 0    # and this

        self.accuracy = 0
        self.experiment_step = 100

    def clear(self):
        self.pointsArray = np.ndarray((0, 3), dtype='float64')
        self.repaint()

    def lr_change(self):
        try:
            self.lr = float(self.lr_edit.text())
        except ValueError:
            self.lr_edit.setText("Invalid value")

    def epochs_change(self):
        try:
            self.epochs_num = int(self.epochs_edit.text())
        except ValueError:
            self.epochs_edit.setText("Invalid value")

    def batch_change(self):
        try:
            new_batch = int(self.batch_edit.text())
            if 0 < new_batch <= len(self.pointsArray):
                self.batch_size = new_batch
            else:
                raise ValueError
        except ValueError:
            self.batch_edit.setText("Invalid value")

    def sizes_change(self):
        try:
            numbers = self.sizes_edit.text().split(",")
            sizes = []
            for num in numbers:
                int_num = int(num)
                if int_num <= 0:
                    raise ValueError
                sizes.append(int_num)
            if len(sizes) < 2 or sizes[0] != 2 or sizes[-1] != 1:
                raise ValueError
            self.sizes = sizes
        except ValueError:
            self.sizes_edit.setText("Invalid value")

    def activ_func_change(self, text: str):
        match text:
            case 'sigmoid':
                self.activation_function = sigma
                self.activ_func_derivative = sigma_derivative
            case 'tanh':
                self.activation_function = np.tanh
                self.activ_func_derivative = tanh_derivative
            case 'ReLU':
                self.activation_function = relu
                self.activ_func_derivative = relu_derivative

    def learn_algorithm_change(self, text: str):
        self.learn_algorithm = text

    def paintEvent(self, e):
        qp = QPainter()
        qp.begin(self)
        qp.fillRect(0, 0, self.width(), self.height(), QBrush(QColor(0, 0, 19)))
        qp.setRenderHint(QPainter.Antialiasing)
        self.drawField(qp)
        self.drawPoints(qp)
        self.drawAccuracy(qp)
        qp.end()

    def drawPoints(self, qp: QPainter) -> None:
        colors = {self.red_value: QColor(250, 22, 22),
                  self.green_value: QColor(129, 240, 2)}
        r = self.pointR
        for x, y, g in self.pointsArray:
            qp.setBrush(colors[g])
            qp.drawEllipse(int((x + 0.5) * self.width()) - r, int((y + 0.5) * self.height()) - r, r << 1, r << 1)

    def drawField(self, qp: QPainter) -> None:
        colors = {self.red_value: QColor(255, 51, 51, 120),
                  self.green_value: QColor(153, 242, 51, 120)}
        interval = self.fieldInterval
        for x in range(len(self.field)):
            for y in range(len(self.field[0])):
                qp.setBrush(colors[self.field[x][y]])
                qp.drawEllipse(x * interval - 2, y * interval - 2, 4, 4)

    def drawAccuracy(self, qp: QPainter) -> None:
        qp.setPen(QColor(247, 243, 7))
        qp.setFont(QFont("Serif", 16))
        qp.fillRect(self.width() // 2 - 71, 20, 190, 40, QColor(0, 0, 0, 155))
        qp.drawText(self.width() // 2 - 65, 50, "Accuracy: {:.2f}".format(self.accuracy))

    def setField(self):
        interval = self.fieldInterval
        fw, fh = len(self.field), len(self.field[0])  # field width and height
        field = np.array([[x * interval / self.width() - 0.5, y * interval / self.height() - 0.5] for x in range(fw) for y in range(fh)])
        self.field = np.round(self.model.evaluate(field)).reshape(fw, fh)

    def calculateAccuracy(self):
        predicts = np.round(self.model.evaluate(self.pointsArray[:, :2]))
        self.accuracy = np.average(predicts == self.pointsArray[:, 2:3])

    def train(self, experiment: bool = False) -> np.ndarray:
        self.model.__init__(self.sizes, self.activation_function)
        layers = self.model.layers
        np.random.shuffle(self.pointsArray)
        batch_s = self.batch_size

        step = self.experiment_step
        cost_array = np.array([0] * (self.epochs_num // step), dtype='float')

        b1, b2, eps, V_dw, V_db, S_dw, S_db = self.initialize_adam()
        iters = 1

        for e in range(self.epochs_num):
            cost_total = 0
            for begin in range(0, len(self.pointsArray) + 1 - batch_s, batch_s):
                batch = self.pointsArray[begin:begin + batch_s]
                x = np.array(batch[:, :2], dtype='float')
                a_l = self.model.evaluate(x)
                a_l = a_l + 0.000000005 - 0.00000001*(a_l > 0.5)   # for numerical stability
                if experiment and e % step == 0:
                    loss = -np.multiply(np.log(a_l), batch[:, 2:3]) - np.multiply(np.log(1-a_l), 1-batch[:, 2:3])
                    cost_total += np.sum(loss)
                da = -np.multiply(1/a_l, batch[:, 2:3]) + np.multiply(1/(1-a_l), 1-batch[:, 2:3])   # loss derivative
                dz = np.multiply(da, (1-a_l)*a_l)
                for l in reversed(range(1, len(layers))):
                    dw = np.matmul(np.transpose(layers[l-1]), dz) / batch_s
                    db = np.sum(dz, axis=0) / batch_s
                    if self.learn_algorithm == 'Adam':
                        V_dw[l-1] = b1 * V_dw[l-1] + (1 - b1) * dw
                        V_db[l-1] = b1 * V_db[l-1] + (1 - b1) * db
                        S_dw[l-1] = b2 * S_dw[l-1] + (1 - b2) * dw*dw
                        S_db[l-1] = b2 * S_db[l-1] + (1 - b2) * db*db
                        V_dw_corrected = V_dw[l-1] / (1 - b1 ** iters)
                        V_db_corrected = V_db[l-1] / (1 - b1 ** iters)
                        S_dw_corrected = S_dw[l-1] / (1 - b2 ** iters)
                        S_db_corrected = S_db[l-1] / (1 - b2 ** iters)
                        dw = V_dw_corrected / (np.sqrt(S_dw_corrected) + eps)
                        db = V_db_corrected / (np.sqrt(S_db_corrected) + eps)
                    da = np.matmul(dz, np.transpose(self.model.weights[l-1]))
                    dz = np.multiply(da, self.activ_func_derivative(layers[l-1]))
                    self.model.weights[l-1] = self.model.weights[l-1] - self.lr * dw
                    self.model.biases[l-1] = self.model.biases[l-1] - self.lr * db
                iters += 1
            if experiment and e % step == 0:
                cost_array[e//step] = cost_total/len(self.pointsArray)

        self.calculateAccuracy()
        self.setField()
        self.repaint()
        if experiment: return cost_array

    def initialize_adam(self):
        b1 = 0.9
        b2 = 0.999
        eps = 0.00000001
        V_dw = np.array([np.zeros(w.shape) for w in self.model.weights], dtype=object)
        V_db = np.array([np.zeros(b.shape) for b in self.model.biases], dtype=object)
        S_dw = np.array([np.zeros(w.shape) for w in self.model.weights], dtype=object)
        S_db = np.array([np.zeros(b.shape) for b in self.model.biases], dtype=object)
        return b1, b2, eps, V_dw, V_db, S_dw, S_db

    def plot_cost_fn(self):
        epoch_num = list(range(0, self.epochs_num, self.experiment_step))
        cost_array = self.train(True)
        plt.plot(epoch_num, cost_array, label=str(self.lr))
        plt.legend()
        plt.show()

    def experiment(self) -> None:
        old_lr = self.lr
        lr_values = [0.01, 0.1, 1]
        epoch_num = list(range(0, self.epochs_num, self.experiment_step))
        for lr in lr_values:
            self.lr = lr
            cost_array = self.train(True)
            plt.plot(epoch_num, cost_array, label=str(lr))
        plt.legend()
        self.lr = old_lr
        plt.show()

    def compare_activ_funcs(self) -> None:
        epoch_num = list(range(0, self.epochs_num, self.experiment_step))
        old_activ, old_derivative = self.activation_function, self.activ_func_derivative
        activation_funcs = [sigma, np.tanh, relu]
        act_func_names = ['sigma', 'tanh', 'relu']
        activation_func_derivatives = [sigma_derivative, tanh_derivative, relu_derivative]
        for func, derivative, name in zip(activation_funcs, activation_func_derivatives, act_func_names):
            self.activation_function = func
            self.activ_func_derivative = derivative
            cost_array = self.train(True)
            plt.plot(epoch_num, cost_array, label=name)
        self.activation_function, self.activ_func_derivative = old_activ, old_derivative
        plt.legend()
        plt.show()

    def compare_learn_algo(self) -> None:
        epoch_num = list(range(0, self.epochs_num, self.experiment_step))
        odl_learn_algo = self.learn_algorithm
        learn_algos = ['Batch GD', 'Adam']
        for learn_algo in learn_algos:
            self.learn_algorithm = learn_algo
            cost_array = self.train(True)
            plt.plot(epoch_num, cost_array, label=learn_algo)
        self.learn_algorithm = odl_learn_algo
        plt.legend()
        plt.show()

    def mousePressEvent(self, e):
        green = self.green_value if e.buttons() & Qt.LeftButton else self.red_value
        self.pointsArray = np.append(self.pointsArray, [[e.x()/self.width() - 0.5, e.y()/self.height() - 0.5, green]], axis=0)
        # self.train()
        self.repaint()


def main():
    app = QApplication(sys.argv)
    win = MyWindow()

    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
