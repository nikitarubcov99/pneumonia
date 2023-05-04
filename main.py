import sys

import cv2
import numpy as np
import tensorflow as tf
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QTableWidget, QFileDialog, QLabel, \
    QPushButton


class Window(QMainWindow, QTableWidget):
    def __init__(self):
        super().__init__()
        # Инициализация основного окна программы и его компонентов
        self.pixmap = None
        self.showDialog = None
        self.file_name = None
        self.acceptDrops()
        self.setWindowTitle("Pneumonia classify")
        self.setGeometry(650, 180, 660, 700)

        # Создание поля для вывода выбранного изображения
        self.label = QtWidgets.QLabel(self)
        self.label.setGeometry(140, 120, 400, 400)
        # Создание поля для вывода результата
        self.label1 = QLabel('Здесь будет отображаться результат', self)
        self.label1.setGeometry(150, 420, 400, 400)
        self.label1.setAlignment(QtCore.Qt.AlignCenter)

        # Добавление кнопки для загрузки фалйа
        self.button = QPushButton("Выбор изображения", self)
        self.button.setGeometry(120, 650, 200, 40)
        self.button1 = QPushButton("Выход", self)
        self.button1.setGeometry(360, 650, 200, 40)
        self.show()

        # Обработка события нажатия на кнопку выполнения
        self.button.clicked.connect(self.get_file_path)
        # Обработка события нажатия на кнопку завершения работы приложения
        self.button1.clicked.connect(self.exit_app)

    def get_file_path(self):
        """
        Метод для получения пути к выбранному для анализа файлу.
        :return:
        """
        self.label.clear()
        file_name = QFileDialog.getOpenFileName(self, 'Open file',
                                                '"C:/Users/nero1/zhenya"')[0]
        # Вызов метода для вывода изображения
        self.print_image(file_name)
        # Вызов метода для классификации пневмонии на изображении
        self.load_model(file_name)

    def print_image(self, file_name):
        """
        Метод для отображения выбранного пользователем изображения.
        :param file_name: указывает путь к выбранному файлу.
        :return:
        """
        self.pixmap = QPixmap(file_name)
        self.pixmap = self.pixmap.scaled(400, 400)

        # Добавление изображения в поле
        self.label.setPixmap(self.pixmap)

    def result_implementation(self, predict_classes):
        """
        Метод для интрпритации численного результата работы нейронной сети.
        :param predict_classes: хранит предсказанный для выбранного изображения класс.
        :return:
        """
        if predict_classes[0] == 0:
            self.label1.setText('Обнаружена пневмония')
        else:
            self.label1.setText('Патологий необнаружено')

    def load_model(self, file_name):
        """
        Метод для загрузки и использования модели, модель загружается в память только при первом вызове метода.
        :param file_name: хранит путь к файлу выбранному пользователем для анализа.
        :return:
        """
        global saved_model, normal_data
        model_k = 0
        self.label1.clear()
        data = []
        img_size = 150

        # Проверка загружена ли модель в память
        if model_k == 0:
            saved_model = tf.keras.models.load_model("pneumonia_classify")
            model_k += 1

        # Предобработка выбранного пользователем изображения для классификации, если файла не существует по пути,
        # вызывается исключение
        try:
            img_arr = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
            resized_arr = cv2.resize(img_arr, (img_size, img_size))
            data.append(resized_arr)
            normal_data = np.array(data)
            normal_data = np.array(normal_data) / 255
            normal_data = normal_data.reshape(-1, img_size, img_size, 1)
        except Exception as e:
            print(e)

        # Использование модели для классификации выбранного пользователем изображения
        predict = saved_model.predict(normal_data)
        predict_classes = np.argmax(predict, axis=1)

        # Вызов метода для интерпритации результатов работы модели
        self.result_implementation(predict_classes)

    @staticmethod
    def exit_app():
        """
        Метод для закрытия приложения.
        :return:
        """
        QApplication.quit()


# Объявления приложения PyQt
App = QApplication(sys.argv)
App.setStyleSheet("QLabel{font-size: 14pt;}")

# Создание экземпляра главного окна программы
window = Window()

# Запуск приложения
sys.exit(App.exec())

