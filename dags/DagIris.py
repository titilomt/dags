from airflow import DAG
from airflow.utils.task_group import TaskGroup
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from datetime import datetime, timedelta

import os
ROOT_DIR = os.path.dirname(__file__)

pathScript = os.path.join(ROOT_DIR, "etl_scripts")
pathModels = os.path.join(ROOT_DIR, "ml_flow")
pathIris = os.path.join(ROOT_DIR, "etl_scripts/featurestore/iris.txt")
pathProcessed = os.path.join(
    ROOT_DIR, "etl_scripts/featurestore/irisProcessed.txt")
pathLog = os.path.join(
    ROOT_DIR, "etl_scripts/log")
pathError = os.path.join(ROOT_DIR, "etl_scripts/log/irisError.txt")


def task_failure_alert(context):
    subject = "[Airflow] DAG {0} - Task {1}: Failed".format(
        context['task_instance_key_str'].split('__')[0],
        context['task_instance_key_str'].split('__')[1]
    )
    html_content = """
    DAG: {0}<br>
    Task: {1}<br>
    Failed on: {2}
    """.format(
        context['task_instance_key_str'].split('__')[0],
        context['task_instance_key_str'].split('__')[1],
        datetime.now()
    )
    #send_email_smtp(dag_vars["dev_mailing_list"], subject, html_content)
    print(subject, html_content)
    text_file = open(pathLog.join('/logs.txt'), "w")
    n = text_file.write(subject + html_content)
    text_file.close()


default_args = {
    'owner': 'admin',
    'depends_on_past': False,
    'start_date': datetime(2019, 1, 1),
    'retries': 0,
    'on_failure_callback': task_failure_alert
}

with DAG(
    'dag-pipeline-iris-v1',
    schedule_interval=timedelta(minutes=1),
    catchup=False,
    default_args=default_args
) as dag:

    start = DummyOperator(task_id="start")

    with TaskGroup("etl", tooltip="etl") as etl:

        t1 = BashOperator(
            dag=dag,
            task_id='download_dataset',
            bash_command="""
            cd {0}/featurestore
            curl -L -o iris.txt  "https://drive.google.com/uc?export=download&id=1rZgVuwYon_3QogTr0-v480PRpi-2l1-v"
            """.format(pathScript)
        )

        [t1]

    with TaskGroup("preProcessing", tooltip="preProcessing") as preProcessing:
        t2 = BashOperator(
            dag=dag,
            task_id='encoder_dataset',
            bash_command="""
            cd {0}/
            python3 etl_preprocessing.py {1} {2} {3}
            """.format(pathScript, pathIris, pathProcessed, pathError)
        )
        [t2]

    with TaskGroup("modelTraining", tooltip="modelTraining") as modelTraining:
        t3 = BashOperator(
            dag=dag,
            task_id='ml_flow_decision_tree',
            bash_command="""
            cd {0}/
            python3 ml_decisionTree.py {1} {2} {3} {4} {5}
            """.format(pathModels, pathProcessed, "IrisClassifierDecisionTree", "ModeloArvoreIris", 20, 3)
        )

        t4 = BashOperator(
            dag=dag,
            task_id='ml_flow_random_forest',
            bash_command="""
            cd {0}/
            python3 ml_randForest.py {1} {2} {3} {4} {5}
            """.format(pathModels, pathProcessed, "IrisClassifierRandomForest", "RandForIris", 1, 6)
        )

        [t3, t4]

    end = DummyOperator(task_id='end')
    start >> etl >> preProcessing >> modelTraining >> end
