# WARNING - THIS IS GENERATED CODE
#   Generator: Eozilla Appligator v0.1.0
#        Date: 2026-05-12T10:37:22.516637

import json
from datetime import datetime

from airflow import DAG
from airflow.models.param import Param
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from kubernetes.client import models as k8s


with DAG(
    dag_id="common-generation",
    start_date=datetime.fromisoformat("2026-05-11"),
    schedule=None,
    catchup=False,
    is_paused_upon_creation=False,
    params={
    "config_path": Param(type='string', title='Config Path', description='Path to the configuration JSON file.')
    },
) as dag:

    tasks = {}


    tasks["common-generation"] = KubernetesPodOperator(
        task_id="common-generation",
        image="quay.io/s2gos/s2gos-mono:0.0.5",
        cmds=["python", "/app/run_step.py"],
        arguments=[json.dumps({
            "func_module": "s2gos_apps.processes.common.generation",
            "func_qualname": "generation",
            "inputs": {"config_path": "{{ params.config_path }}"},
            "output_keys": ['return_value'],
        })],
        env_from=[k8s.V1EnvFromSource(secret_ref=k8s.V1SecretEnvSource(name='s2gos-credentials'))],
        container_resources=k8s.V1ResourceRequirements(requests={'cpu': '2', 'memory': '8Gi'}, limits={'cpu': '4', 'memory': '16Gi'}),
        volumes=[k8s.V1Volume(name='s2gos-output', persistent_volume_claim=k8s.V1PersistentVolumeClaimVolumeSource(claim_name='s2gos-output-pvc')), k8s.V1Volume(name='s2gos-settings', config_map=k8s.V1ConfigMapVolumeSource(name='s2gos-settings'))],
        volume_mounts=[k8s.V1VolumeMount(name='s2gos-output', mount_path='/mnt/output'), k8s.V1VolumeMount(name='s2gos-settings', mount_path='/opt/pixi/', sub_path='s2gos_settings.yaml')],
        do_xcom_push=True,
    )


    def _final_step_callable(ti, upstream_task_id):
        return ti.xcom_pull(task_ids=upstream_task_id)
    
    tasks["__procodile_final_step__"] = PythonOperator(
        task_id="__procodile_final_step__",
        python_callable=_final_step_callable,
        op_kwargs={
            "upstream_task_id": "common-generation"
        },
        do_xcom_push=True
    )

    tasks["common-generation"] >> tasks["__procodile_final_step__"]

