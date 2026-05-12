# WARNING - THIS IS GENERATED CODE
#   Generator: Eozilla Appligator v0.1.0
#        Date: 2026-05-12T10:37:22.534155

import json
from datetime import datetime

from airflow import DAG
from airflow.models.param import Param
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from kubernetes.client import models as k8s


with DAG(
    dag_id="kairouan-generation-config",
    start_date=datetime.fromisoformat("2026-05-11"),
    schedule=None,
    catchup=False,
    is_paused_upon_creation=False,
    params={
    "scene_name": Param(type='string', title='Scene Name', description='Scene id name.'),
    "target_lat": Param(type='number', title='Target Lat', description="Target's center latitude."),
    "target_lon": Param(type='number', title='Target Lon', description="Target's center longitude."),
    "target_size": Param(type='number', title='Target Size', description="Target's size in [km]."),
    "config_output_dir": Param(type='object', description='Generation configuration output directory.', nullable=True, properties={'value': {'type': 'string', 'title': 'Value', 'description': 'Full path URI'}, 'cid': {'type': 'string', 'title': 'Cid', 'description': 'Credential ID', 'nullable': True}}, required=['value']),
    "scene_output_dir": Param(type='object', description='Scene description output directiory.', nullable=True, properties={'value': {'type': 'string', 'title': 'Value', 'description': 'Full path URI'}, 'cid': {'type': 'string', 'title': 'Cid', 'description': 'Credential ID', 'nullable': True}}, required=['value'])
    },
) as dag:

    tasks = {}


    tasks["kairouan-generation-config"] = KubernetesPodOperator(
        task_id="kairouan-generation-config",
        image="quay.io/s2gos/s2gos-mono:0.0.5",
        cmds=["python", "/app/run_step.py"],
        arguments=[json.dumps({
            "func_module": "s2gos_apps.processes.kairouan",
            "func_qualname": "generation_configs",
            "inputs": {"scene_name": "{{ params.scene_name }}",
"target_lat": "{{ params.target_lat }}",
"target_lon": "{{ params.target_lon }}",
"target_size": "{{ params.target_size }}",
"config_output_dir": "{{ params.config_output_dir }}",
"scene_output_dir": "{{ params.scene_output_dir }}"},
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
            "upstream_task_id": "kairouan-generation-config"
        },
        do_xcom_push=True
    )

    tasks["kairouan-generation-config"] >> tasks["__procodile_final_step__"]

