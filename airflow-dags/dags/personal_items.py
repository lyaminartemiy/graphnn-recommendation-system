from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator

def print_hello():
    print("Привет, это мой первый DAG в Airflow!")

# Определяем DAG
with DAG(
    dag_id="personal_items",
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["example"],
) as dag:

    # Создаем задачу
    print_task = PythonOperator(
        task_id="print_hello_task",
        python_callable=print_hello,
    )

    # Если бы было несколько задач, здесь можно было бы определить их порядок
    # Например: task1 >> task2
    # В нашем случае просто оставляем одну задачу

# Альтернативный вариант определения порядка задач (для одной задачи не обязательно)
print_task
