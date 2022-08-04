"""
Code that goes along with the Airflow tutorial located at:
https://github.com/apache/airflow/blob/master/airflow/example_dags/tutorial.py
"""
from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
import generate_graphs_remote as gr
from pprint import pprint
import logging
import os


path_var = "/home/srijith/.tej/jobs/attn_pytorch/stage"
source_var = "attention_pytorch.py"
env_var = "pytorch_env"


def print_context(ds, **kwargs):
    pprint(kwargs)
    print(ds)
    env = kwargs["env"]
    path = kwargs["path"]
    filename = kwargs["filename"]
    file_out = kwargs["filename_out"]
    logfile = kwargs["logfile"]
    args = kwargs.get("args", "")

    cmd = (
        "source activate "
        + env
        + " && cd "
        + path
        + " && python "
        + filename
        + " "
        + args
        + " > "
        + file_out
    )

    f = open(logfile, "a")
    f.write(str(datetime.now()) + "   " + str(cmd) + "\n")
    f.close()

    cmd2 = '/bin/bash -c "' + cmd + '"'
    os.system(cmd2)

    return "Whatever you return gets printed in the logs"


def generate_graphs(ds, **kwargs):
    gr.get_history_single(
        "/home/srijith/.tej/jobs/attn_pytorch/stage/model_metrics_charngrams.json",
        "/home/srijith/.tej/jobs/attn_pytorch/stage/history.pdf",
        "Training and validation accuracy for Charngrams",
    )
    gr.get_boxplots_single(
        "/home/srijith/.tej/jobs/attn_pytorch/stage/model_metrics_charngrams.json",
        "/home/srijith/.tej/jobs/attn_pytorch/stage/metrics_boxplot.pdf",
        "Metrics for Charngrams",
    )
    return 0


default_args = {
    "owner": "srijith",
    "depends_on_past": False,
    "start_date": datetime(2019, 6, 27),
    "email": ["srijithr@vt.edu"],
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
    "catchup": False,
    # 'queue': 'bash_queue',
    # 'pool': 'backfill',
    # 'priority_weight': 10,
    # 'end_date': datetime(2016, 1, 1),
}

dag = DAG("pytorch_rnn", default_args=default_args, schedule_interval=None)


rnn_run = PythonOperator(
    task_id="rnn_run",
    provide_context=True,
    python_callable=print_context,
    op_kwargs={
        "filename_out": "out",
        "logfile": "log",
        "filename": source_var,
        "path": path_var,
        "env": env_var,
    },
    dag=dag,
)


rnn_run2 = PythonOperator(
    task_id="rnn_run2",
    provide_context=True,
    python_callable=print_context,
    op_kwargs={
        "filename_out": "out2",
        "logfile": "log",
        "filename": source_var,
        "path": path_var,
        "env": env_var,
    },
    dag=dag,
)


plot_run = PythonOperator(
    task_id="plot_run",
    provide_context=True,
    python_callable=generate_graphs,
    op_kwargs={
        "filename_out": "out",
        "logfile": "log",
        "filename": source_var,
        "path": path_var,
        "env": env_var,
    },
    dag=dag,
)

remove_model_metrics = BashOperator(
    task_id="remove_model_metrics",
    bash_command="cd /home/srijith/.tej/jobs/attn_pytorch/stage && rm model_metrics.json && rm roc_metrics.json",
    dag=dag,
)


cp_model_metrics = BashOperator(
    task_id="cp_model_metrics",
    bash_command="cd /home/srijith/.tej/jobs/attn_pytorch/stage && cp model_metrics.json model_metrics_charngrams.json",
    dag=dag,
)

remove_model_metrics >> rnn_run >> rnn_run2 >> cp_model_metrics >> plot_run
