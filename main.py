from tkinter import *
from tkinter.scrolledtext import ScrolledText
from tkinter import ttk
from tkinter.messagebox import showinfo
from tkinter import filedialog
from tensorflow import keras
import pandas as pd
from src.users import users
import os


def load_model():
    '''
    Загрузка модели
    Автор: Стасевич Екатерина Алексеевна
    Тема ВКР: Прогнозирование количества отчисленных студентов с помощью методов машинного обучения
    '''
    # Возвращает модель
    return keras.models.load_model('model.keras')


# Интерфейс приложения
def open_app():
    # Сохраняет модель в переменную
    model = load_model()
    # Логика работы приложения
    def predict():
        # Получить данные из поля ввода (путь к файлу)
        file_path = data_entry.get()

        # Обработка отсутствия файла по введенному адресу
        if not os.path.exists(file_path):
            showinfo(title='404', message='Ошибка! Файл с таким именем не найден. \n'
                                          'Пожалуйста, проверьте правильность указанного пути.')
            return

        # Обработка ошибки чтения файла
        try:
            data = pd.read_csv(file_path, dtype={'id_student': str})
        except Exception:
            showinfo(title='Ошибка чтения файла', message='Не удалось прочитать файл')
            return

        try:
            id_list = data['id_student'] # получаем список "имен"
            X_new = data.drop(columns=['id_student']) # удаляем ненужную колонку
        except KeyError:
            showinfo(title='Ошибка чтения файла', message='Неверное содержание файла')

        # Предсказание, вероятность попадания в группу риска
        predicts = model.predict(X_new)
        # Представление класса 1 или 0
        pre_res = (predicts.flatten() > 0.5).astype(int)

        total = len(pre_res)
        dropout = pre_res.sum()
        percent = dropout / total * 100

        # Вывод результата
        result_label.config(text=f'Из {total} студентов риск отчисления у {dropout}.\n'
                                 f'Группа риска составляет {percent:.2f} % от общего числа')

        result = pd.DataFrame({
            'id_student': id_list,
            'predicts': predicts.flatten(),
            'result': pre_res
        })

        # Сортировка только группы риска по убыванию
        risk_group = result[result['result'] == 1].sort_values('predicts', ascending=False)

        # Вставка заголовка
        result_entry.delete('1.0', 'end')
        result_entry.insert('end', 'Студенты в группе риска:\n\n')

        # Итерированная вставка списка студентов
        for _, i in risk_group.iterrows():
            result_entry.insert('end', f"{i['id_student']}, вероятность отчисления: {i['predicts']:.1%}\n")

    # Выбор файла с компьютера
    def load_file():
        initial_dir = os.getcwd()  # получить текущую папку
        filepath = filedialog.askopenfilename(initialdir=initial_dir, defaultextension='csv',
                                              filetypes=[('CSV', '*.csv')])  # только csv
        data_entry.delete(0, END)  # очистить поле
        data_entry.insert(0, filepath)  # ввести путь к файлу

    # Запись результатов предсказаний
    def write_csv():
        text = result_entry.get('1.0', 'end')

        with open('risk_group.txt', 'w', encoding='utf-8') as f:
            f.write(text)

        showinfo(title='Сохранено', message='Результаты сохранены в risk_group.txt')


    def clear_result_entry():
        result_entry.delete('1.0', 'end')
        result_label.config(text='')

    root = Tk()
    # Заголовок окна
    root.title('Прогнозирование отчисленных студентов')
    # Фиксированные размеры окна
    root.geometry('1000x500')
    root.resizable(False, False)

    # Главный фрейм для ввода данных и вывода основного результата
    main_frame = Frame(root, width=600, height=500, background='#f6f5fb')
    main_frame.pack(side='left', fill='both', expand=True)

    # Дополнительный фрейм для списка студентов
    result_frame = Frame(root, width=400, height=500, background='#f6f5fb')
    result_frame.pack(side='right', fill='both', expand=True)

    # Подпись для окошка ввода имени файла
    istr_label = ttk.Label(main_frame,
                       text='Загрузите данные для анализа в формате .csv',
                       font=('Arial', 14), background='#f6f5fb')
    istr_label.pack(anchor='w', pady=10, padx=10)

    # Фрейм для поля ввода и кнопки, чтобы поставить их рядом
    file_frame = Frame(main_frame, background='#f6f5fb')
    file_frame.pack(fill='x', pady=(0, 20))

    # Окно ввода имени файла
    data_entry = ttk.Entry(file_frame, font='20', width=50)
    data_entry.insert(0, 'datasets/new_data.csv')
    data_entry.pack(side='left', fill='x', expand=True, padx=10, pady=10)

    # Кнопка для выбора файла
    rewiew_btn = Button(file_frame, text='Обзор', command=load_file, background='#f6f5fb')
    rewiew_btn.pack(side='right')

    # Кпопка для расчета
    pred_btn = Button(main_frame, text='Рассчитать', font=20, background='#B3AADF',
                      command=predict) # привязка к функции
    pred_btn.pack()

    # Блок для вывода результата
    result_label = ttk.Label(main_frame, text='',
                             font=('Arial', 14), background='#f6f5fb')
    result_label.pack(anchor='w', pady=10, padx=10)

    # Поле вывода списка студентов с результатами
    result_entry = ScrolledText(result_frame, width=400, height=500,
                                font=('Arial', 14))
    result_entry.pack(anchor='w', pady=10, padx=10)

    bottom_frame = Frame(main_frame, background='#f6f5fb')
    bottom_frame.pack(side='right', fill='y', anchor='s')

    # Кнопка для сохранения результатов
    res_btn = Button(bottom_frame, text='Сохранить данные', font=20, background='#B3AADF',
                     command=write_csv, width=20)
    res_btn.pack(side='bottom', pady=(0, 5))

    # Кнопка для очистки вывода
    clear_btn = Button(bottom_frame, text='Очистить поля вывода', font=20, background='#B3AADF',
                       command=clear_result_entry, width=20)
    clear_btn.pack(side='bottom')

    root.mainloop()
    return root     # возвращаем окно для дальнейшего использования в predict()


# Успешный вход
def login_success():
    login_window.destroy()
    open_app()


def login_wrong():
    showinfo(title='Ошибка входа', message='Неверное имя пользователя или пароль')


'''
Проверка имени и пароля пользователя
Получает строки из полей ввода: 
    name - имя пользователя из поля username_entry;
    password - пароль из поля password_entry
'''
def user_check():
    # Получает данные из полей ввода:
    name = username_entry.get()
    password = password_entry.get()

    # Проверяет совпадение логина и пароля
    if users.get(name) == password:
        login_success()     # успешный вход
    else:
        login_wrong()   # ошибка входа


login_window = Tk()

# Фиксированные размеры окна
login_window.geometry('1000x500')
login_window.resizable(False, False)

login_window.title('Вход – Прогнозирование отчисленных студентов')
login_window.configure(background='#f6f5fb')

# Поле ввода имени пользователя
label_name = Label(login_window, text='Введите имя пользователя',
                   font='20', width=50, background='#f6f5fb')
username_entry = Entry(login_window, font='20', width=50)
label_name.pack(padx=10, pady=10)
username_entry.pack(padx=10, pady=10)

# Поле ввода пароля
label_password = Label(login_window, text='Введите пароль',
                       font='20', width=50, background='#f6f5fb')
password_entry = Entry(login_window, font='20', width=50, show='*') # закрывает символы *
label_password.pack(padx=10, pady=10)
password_entry.pack(padx=10, pady=10)

# Кнопка "Вход"
entered_btn = Button(login_window, text='Войти в систему', font='20', command=user_check, background='#B3AADF')
entered_btn.pack(padx=10, pady=10)

login_window.mainloop()
