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
    "pytorch_attention_rnn_runner", default_args=default_args, schedule_interval=None
)

# t1, t2 and t3 are examples of tasks created by instantiating operators
t1 = BashOperator(
    task_id="Pytorch_attention",
    # bash_command='source activate pytorch_env && cd /home/srijith/.tej/jobs/attn_pytorch/stage && python Pytorch_attention.py',
    bash_command="source activate pytorch_env && cd /home/srijith/.tej/jobs/attn_pytorch/stage && python attention_pytorch.py > out",
    dag=dag,
)

t2 = BashOperator(
    task_id="Pytorch_attention2",
    # bash_command='source activate pytorch_env && cd /home/srijith/.tej/jobs/attn_pytorch/stage && python Pytorch_attention.py',
    bash_command="source activate pytorch_env && cd /home/srijith/.tej/jobs/attn_pytorch/stage && python attention_pytorch.py > out2",
    dag=dag,
)

t3 = BashOperator(
    task_id="Pytorch_attention3",
    # bash_command='source activate pytorch_env && cd /home/srijith/.tej/jobs/attn_pytorch/stage && python Pytorch_attention.py',
    bash_command="source activate pytorch_env && cd /home/srijith/.tej/jobs/attn_pytorch/stage && python attention_pytorch.py > out3",
    dag=dag,
)

t4 = BashOperator(
    task_id="Pytorch_attention4",
    # bash_command='source activate pytorch_env && cd /home/srijith/.tej/jobs/attn_pytorch/stage && python Pytorch_attention.py',
    bash_command="source activate pytorch_env && cd /home/srijith/.tej/jobs/attn_pytorch/stage && python attention_pytorch.py > out4",
    dag=dag,
)

t5 = BashOperator(
    task_id="Pytorch_attention5",
    # bash_command='source activate pytorch_env && cd /home/srijith/.tej/jobs/attn_pytorch/stage && python Pytorch_attention.py',
    bash_command="source activate pytorch_env && cd /home/srijith/.tej/jobs/attn_pytorch/stage && python attention_pytorch.py > out5",
    dag=dag,
)

t6 = BashOperator(
    task_id="model_metrics",
    bash_command='cd /home/srijith/.tej/jobs/attn_pytorch/stage && cp model_metrics.json {{ dag_run.conf["metrics_file"]  if dag_run else "" }}',
    # bash_command='cd /home/srijith/.tej/jobs/attn_pytorch/stage && cp model_metrics.json model_metrics_charngrams.json',
    dag=dag,
)

t7 = BashOperator(
    task_id="test_param",
    bash_command='echo "hello" > out',
    # bash_command='cd /home/srijith/.tej/jobs/attn_pytorch/stage && cp model_metrics.json model_metrics_charngrams.json',
    dag=dag,
)

t5 >> t7
