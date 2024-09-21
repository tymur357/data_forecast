import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

# Налаштування палітри
color_pal = sns.color_palette()
plt.style.use('fivethirtyeight')

# Зчитування даних
data = pd.read_csv('Data.csv')
data = data.set_index('Date')
data.index = pd.to_datetime(data.index)

# Розбиття даних
data_separator = '01-01-2015'
train = data.loc[data.index < data_separator]
test = data.loc[data.index >= data_separator]

# Розділення даних для аналізу
def create_columns(data):
    data = data.copy()
    data['hour'] = data.index.hour
    data['dayofweek'] = data.index.dayofweek
    data['quarter'] = data.index.quarter
    data['month'] = data.index.month
    data['year'] = data.index.year
    data['dayofyear'] = data.index.dayofyear
    data['dayofmonth'] = data.index.day
    data['weekofyear'] = data.index.isocalendar().week
    return data

data = create_columns(data)

# Створення моделі для навчання
train = create_columns(train)
test = create_columns(test)

COLUMNS = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month', 'year']
POWER = 'Power_Consumed'

X_train = train[COLUMNS]
y_train = train[POWER]

X_test = test[COLUMNS]
y_test = test[POWER]

regression_model = xgb.XGBRegressor(base_score=0.5, booster='gbtree',
                                    n_estimators=1000,
                                    early_stopping_rounds=50,
                                    objective='reg:linear',
                                    max_depth=3,
                                    learning_rate=0.01)
regression_model.fit(X_train, y_train,
                    eval_set=[(X_train, y_train), (X_test, y_test)],
                    verbose=100)

# Прогноз даних
test['prediction'] = regression_model.predict(X_test)
data = data.merge(test[['prediction']], how='left', left_index=True, right_index=True)

# Прогноз даних на майбутнє
future_data_prediction_limit = '2018-12-25'
future = pd.date_range(data.index.max(), future_data_prediction_limit, freq='1h')
future_data = pd.DataFrame(index=future)
future_data['isFuture'] = True
data['isFuture'] = False
data_and_future_data = pd.concat([data, future_data])
data_and_future_data = create_columns(data_and_future_data)
future_new_columns = data_and_future_data.query('isFuture').copy()
future_new_columns['pred'] = regression_model.predict(future_new_columns[COLUMNS])

# Головне вікно
class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.content()

    def content(self):
        # Вікно
        self.setWindowTitle("PowerForecast")
        self.move(300, 300)
        self.resize(2000, 1000)

        # Графік
        fig, ax = plt.subplots(figsize=(15, 5))
        data['Power_Consumed'].plot(ax=ax, style='.', color=sns.color_palette()[0],
                            title='Використання енергії в МВт')
        # Ox, Oy
        ax.set_xlabel('Роки')
        ax.set_ylabel('МВт')
        ax.legend(['Кількість спожитих МВт/год'])

        # Розміщення
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        layout = QVBoxLayout(self.central_widget)

        # Кнопки
        button_layout = QHBoxLayout()
        self.open_split_button = QPushButton("Розділення даних")
        font = QFont()
        font.setPointSize(18)
        self.open_split_button.setFont(font)
        self.open_split_button.clicked.connect(self.open_split)
        self.open_lower_scale_button = QPushButton("Дані в меншому масштабі")
        self.open_lower_scale_button.setFont(font)
        self.open_lower_scale_button.clicked.connect(self.open_lower_scale)
        button_layout.addWidget(self.open_split_button)
        button_layout.addWidget(self.open_lower_scale_button)
        layout.addLayout(button_layout)

        button_layout_2 = QHBoxLayout()
        self.open_hours_button = QPushButton("Розподілення даних за годинами")
        self.open_hours_button.setFont(font)
        self.open_hours_button.clicked.connect(self.open_hours)
        self.open_day_button = QPushButton("Розподілення даних за днями тижня")
        self.open_day_button.setFont(font)
        self.open_day_button.clicked.connect(self.open_day)
        button_layout_2.addWidget(self.open_hours_button)
        button_layout_2.addWidget(self.open_day_button)
        layout.addLayout(button_layout_2)

        button_layout_3 = QHBoxLayout()
        self.open_month_button = QPushButton("Розподілення даних за місяцями")
        self.open_month_button.setFont(font)
        self.open_month_button.clicked.connect(self.open_month)
        self.open_year_button = QPushButton("Розподілення даних за роками")
        self.open_year_button.setFont(font)
        self.open_year_button.clicked.connect(self.open_year)
        button_layout_3.addWidget(self.open_month_button)
        button_layout_3.addWidget(self.open_year_button)
        layout.addLayout(button_layout_3)

        button_layout_4 = QHBoxLayout()
        self.open_usability_button = QPushButton("Корисність даних")
        self.open_usability_button.setFont(font)
        self.open_usability_button.clicked.connect(self.open_data_usability)
        self.open_predictions_button = QPushButton("Прогноз")
        self.open_predictions_button.setFont(font)
        self.open_predictions_button.clicked.connect(self.open_predictions)
        button_layout_4.addWidget(self.open_usability_button)
        button_layout_4.addWidget(self.open_predictions_button)
        layout.addLayout(button_layout_4)

        button_layout_5 = QHBoxLayout()
        self.lower_scale_predictions_button = QPushButton("Детальне порівняння")
        self.lower_scale_predictions_button.setFont(font)
        self.lower_scale_predictions_button.clicked.connect(self.open_lower_scale_predictions)
        self.open_predictions_future_button = QPushButton("Прогноз на майбутнє")
        self.open_predictions_future_button.setFont(font)
        self.open_predictions_future_button.clicked.connect(self.open_future_predictions)
        button_layout_5.addWidget(self.lower_scale_predictions_button)
        button_layout_5.addWidget(self.open_predictions_future_button)
        layout.addLayout(button_layout_5)

        self.canvas = FigureCanvas(fig)
        layout.addWidget(self.canvas)

        # Зазначення ©
        lbl_x = 0.30
        lbl_y = -0.20
        ax.annotate('©Програму розробив студент групи КН-413 Умрихін Тимур',
                    xy=(lbl_x, lbl_y),
                    xycoords='axes fraction', fontsize=15)

    # Функції для інших вікон
    def open_split(self):
        new_window = Split(self)
        new_window.show()

    def open_lower_scale(self):
        new_window = LowerScale(self)
        new_window.show()

    def open_hours(self):
        new_window = Hours(self)
        new_window.show()

    def open_day(self):
        new_window = Days(self)
        new_window.show()

    def open_month(self):
        new_window = Month(self)
        new_window.show()

    def open_year(self):
        new_window = Year(self)
        new_window.show()

    def open_data_usability(self):
        new_window = Usability(self)
        new_window.show()

    def open_predictions(self):
        new_window = Predictions(self)
        new_window.show()

    def open_lower_scale_predictions(self):
        new_window = LowerScalePredictions(self)
        new_window.show()

    def open_future_predictions(self):
        new_window = FutureData(self)
        new_window.show()

# Вікно з розділенням даних
class Split(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.content()

    def content(self):
        # Вікно
        self.setWindowTitle("PowerForecast")
        self.move(300, 300)
        self.resize(2000, 900)

        # Графік
        fig, ax = plt.subplots(figsize=(15, 5))
        train['Power_Consumed'].plot(ax=ax, label='Training Set', title='Data Splitting')
        test['Power_Consumed'].plot(ax=ax, label='Test Set')
        ax.axvline(data_separator, color='black')

        # Ox, Oy
        ax.set_xlabel('Роки')
        ax.set_ylabel('МВт')
        ax.legend(['Дані для тренування', 'Дані для тесту'])

        # Розміщення
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        layout = QVBoxLayout(self.central_widget)
        self.canvas = FigureCanvas(fig)
        layout.addWidget(self.canvas)

        # Зазначення ©
        lbl_x = 0.30
        lbl_y = -0.18
        ax.annotate('©Програму розробив студент групи КН-413 Умрихін Тимур',
                    xy=(lbl_x, lbl_y),
                    xycoords='axes fraction', fontsize=15)

# Вікно з масштабуванням даних
class LowerScale(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.content()

    def content(self):
        # Вікно
        self.setWindowTitle("PowerForecast")
        self.setGeometry(300, 300, 1200, 800)
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Поле для даних
        date_range_input = QLineEdit(self)
        date_range_input.setPlaceholderText("Введіть проміжок: Приклад ('01-06-2004' до '01-12-2004')")
        layout.addWidget(date_range_input)

        # Графік
        fig, ax = plt.subplots(figsize=(15, 5))

        # Оновлення даних і побудова графіка
        def update_plot():
            date_range = date_range_input.text()
            start_date, end_date = date_range.split(' до ')
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)

            data_filtered = data.loc[(data.index >= start_date) & (data.index <= end_date)]
            data_filtered = data_filtered.interpolate()
            data_filtered = data_filtered.sort_index()

            ax.clear()
            data_filtered['Power_Consumed'].plot(ax=ax, style='-')
            ax.set_xlabel('Час')
            ax.set_ylabel('МВт')
            ax.set_title('Використання енергії в МВт')
            ax.legend(['Кількість спожитих МВт/год'])

            lbl_x = 0.17
            lbl_y = -0.25
            ax.annotate('©Програму розробив студент групи КН-413 Умрихін Тимур', xy=(lbl_x, lbl_y),
                        xycoords='axes fraction', fontsize=15)
            canvas.draw()

        date_range_input.returnPressed.connect(update_plot)
        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)

        # Дефолтні значення
        default_date_range = '01-06-2004 до 01-12-2004'
        date_range_input.setText(default_date_range)
        update_plot()


# Вікно з розділенням даних по годинах
class Hours(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.content()

    def content(self):
        # Вікно
        self.setWindowTitle("PowerForecast")
        self.move(300, 300)
        self.resize(2000, 1000)

        # Графік
        fig, ax = plt.subplots(figsize=(15, 8))
        sns.boxplot(data=data, x='hour', y='Power_Consumed')
        ax.set_title('МВт на год')

        # Ox, Oy
        ax.set_xlabel('Година')
        ax.set_ylabel('МВт')
        new_labels = [str(i) for i in range(1, 25)]
        ax.set_xticks(range(24))
        ax.set_xticklabels(new_labels)

        # Розміщення
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        layout = QVBoxLayout(self.central_widget)
        self.canvas = FigureCanvas(fig)
        layout.addWidget(self.canvas)

        # Зазначення ©
        lbl_x = 0.65
        lbl_y = -0.06
        ax.annotate('©Програму розробив студент групи КН-413 Умрихін Тимур',
                    xy=(lbl_x, lbl_y),
                    xycoords='axes fraction', fontsize=15)

# Вікно з розділенням даних по днях
class Days(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.content()
    def content(self):
        # Вікно
        self.setWindowTitle("PowerForecast")
        self.move(300, 300)
        self.resize(2000, 1000)

        # Графік
        fig, ax = plt.subplots(figsize=(15, 8))
        sns.boxplot(data=data, x='dayofweek', y='Power_Consumed')
        ax.set_title('МВт на день')

        # Ox, Oy
        ax.set_xlabel('День тижня')
        ax.set_ylabel('МВт')
        new_labels = [str(i) for i in range(1, 8)]
        ax.set_xticks(range(7))
        ax.set_xticklabels(new_labels)

        # Розміщення
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        layout = QVBoxLayout(self.central_widget)
        self.canvas = FigureCanvas(fig)
        layout.addWidget(self.canvas)

        # Зазначення ©
        lbl_x = 0.65
        lbl_y = -0.06
        ax.annotate('©Програму розробив студент групи КН-413 Умрихін Тимур',
                    xy=(lbl_x, lbl_y),
                    xycoords='axes fraction', fontsize=15)

# Вікно з розділенням даних по місяцях
class Month(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.content()
    def content(self):
        # Вікно
        self.setWindowTitle("PowerForecast")
        self.move(300, 300)
        self.resize(2000, 1000)

        # Графік
        fig, ax = plt.subplots(figsize=(15, 8))
        sns.boxplot(data=data, x='month', y='Power_Consumed')
        ax.set_title('МВт на місяць')

        # Ox, Oy
        ax.set_xlabel('Місяць')
        ax.set_ylabel('МВт')
        new_labels = [str(i) for i in range(1, 13)]
        ax.set_xticks(range(12))
        ax.set_xticklabels(new_labels)

        # Розміщення
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        layout = QVBoxLayout(self.central_widget)
        self.canvas = FigureCanvas(fig)
        layout.addWidget(self.canvas)

        # Зазначення ©
        lbl_x = 0.65
        lbl_y = -0.06
        ax.annotate('©Програму розробив студент групи КН-413 Умрихін Тимур',
                xy=(lbl_x, lbl_y),
                xycoords='axes fraction', fontsize=15)

# Вікно з розділенням даних по роках
class Year(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.content()

    def content(self):
        # Вікно
        self.setWindowTitle("PowerForecast")
        self.move(300, 300)
        self.resize(2000, 1000)

        # Графік
        fig, ax = plt.subplots(figsize=(15, 8))
        sns.boxplot(data=data, x='year', y='Power_Consumed')
        ax.set_title('МВт на рік')

        # Ox, Oy
        ax.set_xlabel('Рік')
        ax.set_ylabel('МВт')

        # Розміщення
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        layout = QVBoxLayout(self.central_widget)
        self.canvas = FigureCanvas(fig)
        layout.addWidget(self.canvas)

        # Зазначення ©
        lbl_x = 0.65
        lbl_y = -0.06
        ax.annotate('©Програму розробив студент групи КН-413 Умрихін Тимур',
                    xy=(lbl_x, lbl_y),
                    xycoords='axes fraction', fontsize=15)


# Вікно з розділенням даних по важливості
class Usability(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.content()

    def content(self):
        # Вікно
        self.setWindowTitle("PowerForecast")
        self.move(300, 300)
        self.resize(2000, 1000)

        # Графік
        fi = pd.DataFrame(data=regression_model.feature_importances_,
        index=COLUMNS,
        columns=['importance'])
        fi.sort_values('importance', inplace=True)
        fig = Figure()
        ax = fig.add_subplot(111)
        fi.plot(kind='barh', title='Важливість атрибута', legend=False, ax=ax)

        # Ox, Oy
        ax.set_xlabel('Важливість атрибута')
        ax.set_ylabel('Атрибут')

        # Розміщення
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        layout = QVBoxLayout(self.central_widget)
        self.canvas = FigureCanvas(fig)
        layout.addWidget(self.canvas)

        # Зазначення ©
        lbl_x = 0.65
        lbl_y = -0.06
        ax.annotate('©Програму розробив студент групи КН-413 Умрихін Тимур',
                    xy=(lbl_x, lbl_y),
                    xycoords='axes fraction', fontsize=15)


# Вікно з прогнозом
class Predictions(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.content()
    def content(self):
        # Вікно
        self.setWindowTitle("PowerForecast")
        self.setGeometry(300, 300, 2000, 1000)

        # Графік
        fig, ax = plt.subplots(figsize=(15, 5))
        data[['Power_Consumed']].plot(ax=ax)
        data['prediction'].plot(ax=ax, style='.')
        plt.legend(['Кількість спожитих МВт/год', 'Кількість спрогнозованих МВт/год()'])
        ax.set_title('Спрогнозовані значення')

        # Ox, Oy
        ax.set_xlabel('Роки')
        ax.set_ylabel('МВт')
        ax.legend(['Кількість спожитих МВт/год', 'Кількість спрогнозованих МВт/год()'])

        # Розміщення
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        layout = QVBoxLayout(self.central_widget)
        self.canvas = FigureCanvas(fig)
        layout.addWidget(self.canvas)

        # Зазначення ©
        lbl_x = 0.30
        lbl_y = -0.15
        ax.annotate('©Програму розробив студент групи КН-413 Умрихін Тимур',
                    xy=(lbl_x, lbl_y),
                    xycoords='axes fraction', fontsize=15)


# Вікно з детальним масштабуванням даних
class LowerScalePredictions(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.content()

    def content(self):
        # Вікно
        self.setWindowTitle("PowerForecast")
        self.setGeometry(300, 300, 2000, 1000)
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Поле для даних
        date_range_input = QLineEdit(self)
        date_range_input.setPlaceholderText("Введіть проміжок: Приклад ('01-01-2016' до '01-04-2016')")
        layout.addWidget(date_range_input)

        # Графік
        fig, ax = plt.subplots(figsize=(15, 5))

        # Оновлення даних і побудова графіка
        def update_plot():
            date_range = date_range_input.text()
            start_date, end_date = date_range.split(' до ')

            data_filtered = data.loc[(data.index >= start_date) & (data.index <= end_date)]
            data_filtered = data_filtered.interpolate()
            data_filtered = data_filtered.sort_index()

            ax.clear()
            data_filtered['Power_Consumed'].plot(ax=ax)
            data_filtered['prediction'].plot(ax=ax, style='.')
            plt.legend(['Кількість спожитих МВт/год', 'Спрогнозовані дані'])
            ax.set_xlabel('Час')
            ax.set_ylabel('МВт')
            ax.set_title('Використання енергії в МВт')
            lbl_x = 0.17
            lbl_y = -0.25
            ax.annotate('©Програму розробив студент групи КН-413 Умрихін Тимур', xy=(lbl_x, lbl_y),
                        xycoords='axes fraction', fontsize=15)
            canvas.draw()

        date_range_input.returnPressed.connect(update_plot)
        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)

        # Дефолтні значення
        default_date_range = '01-01-2016 до 01-04-2016'
        date_range_input.setText(default_date_range)
        update_plot()


# Вікно з майбутніми даними
class FutureData(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.content()

    def content(self):
        # Вікно
        self.setWindowTitle("PowerForecast")
        self.move(300, 300)
        self.resize(2000, 1000)

        # Графік
        fig, ax = plt.subplots(figsize=(15, 5))
        future_new_columns['pred'].plot(ax=ax, color=sns.color_palette()[1],
        lw=1, title='Використання енергії в МВт')

        # Ox, Oy
        ax.set_xlabel('Роки')
        ax.set_ylabel('МВт')
        ax.legend(['Кількість спожитих МВт/год'])

        # Розміщення
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        layout = QVBoxLayout(self.central_widget)
        self.canvas = FigureCanvas(fig)
        layout.addWidget(self.canvas)
        self.addToolBar(NavigationToolbar(self.canvas, self))

        # Зазначення ©
        lbl_x = 0.60
        lbl_y = -0.09
        ax.annotate('©Програму розробив студент групи КН-413 Умрихін Тимур',
                    xy=(lbl_x, lbl_y),
                    xycoords='axes fraction', fontsize=15)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())