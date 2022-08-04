"""
Code that goes along with the Airflow tutorial located at:
https://github.com/apache/airflow/blob/master/airflow/example_dags/tutorial.py
"""
from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from datetime import datetime, timedelta


default_args = {
    "owner": "srijith",
    "depends_on_past": False,
    "start_date": datetime(2019, 6, 27),
    "email": ["srijithr@vt.edu"],
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
    # 'queue': 'bash_queue',
    # 'pool': 'backfill',
    # 'priority_weight': 10,
    # 'end_date': datetime(2016, 1, 1),
}

dag = DAG(
    "pytorch_elmo_runner",
    default_args=default_args,
    schedule_interval=timedelta(days=1),
)

# t1, t2 and t3 are examples of tasks created by instantiating operators
t1 = BashOperator(
    task_id="Pytorch_elmo",
    # bash_command='source activate pytorch_env && cd /home/srijith/.tej/jobs/Pytorch_curated/stage && python Pytorch_elmo.py',
    bash_command="source activate pytorch_env && cd /home/srijith/.tej/jobs/Pytorch_curated/stage && python Elmo_pytorch.py > out",
    dag=dag,
)

t2 = BashOperator(
    task_id="Pytorch_elmo2",
    # bash_command='source activate pytorch_env && cd /home/srijith/.tej/jobs/Pytorch_curated/stage && python Pytorch_elmo.py',
    bash_command="source activate pytorch_env && cd /home/srijith/.tej/jobs/Pytorch_curated/stage && python Elmo_pytorch.py > out2",
    dag=dag,
)

t3 = BashOperator(
    task_id="Pytorch_elmo3",
    # bash_command='source activate pytorch_env && cd /home/srijith/.tej/jobs/Pytorch_curated/stage && python Pytorch_elmo.py',
    bash_command="source activate pytorch_env && cd /home/srijith/.tej/jobs/Pytorch_curated/stage && python Elmo_pytorch.py > out3",
    dag=dag,
)

t4 = BashOperator(
    task_id="Pytorch_elmo4",
    # bash_command='source activate pytorch_env && cd /home/srijith/.tej/jobs/Pytorch_curated/stage && python Pytorch_elmo.py',
    bash_command="source activate pytorch_env && cd /home/srijith/.tej/jobs/Pytorch_curated/stage && python Elmo_pytorch.py > out4",
    dag=dag,
)

t5 = BashOperator(
    task_id="Pytorch_elmo5",
    # bash_command='source activate pytorch_env && cd /home/srijith/.tej/jobs/Pytorch_curated/stage && python Pytorch_elmo.py',
    bash_command="source activate pytorch_env && cd /home/srijith/.tej/jobs/Pytorch_curated/stage && python Elmo_pytorch.py > out5",
    dag=dag,
)

t1 >> t2 >> t3 >> t4 >> t5
