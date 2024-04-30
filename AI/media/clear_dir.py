import os

def delete_files_in_directory(directory_path):
    # Отримуємо список файлів у директорії
    files = os.listdir(directory_path)

    # Видаляємо кожен файл у цьому списку
    for file_name in files:
        file_path = os.path.join(directory_path, file_name)
        try:
            os.remove(file_path)
            print(f"Файл {file_name} був успішно видалений")
        except Exception as e:
            print(f"Помилка при видаленні файлу {file_name}: {e}")

# Приклад використання функції:
directory_path = r'C:\Users\PC\PycharmProjects\lab1\AI\res\TestModel\cars'
delete_files_in_directory(directory_path)
directory_path = r'C:\Users\PC\PycharmProjects\lab1\AI\res\TestModel\no_cars'
delete_files_in_directory(directory_path)
