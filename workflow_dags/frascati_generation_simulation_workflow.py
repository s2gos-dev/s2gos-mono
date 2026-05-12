# WARNING - THIS IS GENERATED CODE
#   Generator: Eozilla Appligator v0.1.0
#        Date: 2026-05-12T11:18:21.672352

import json
from datetime import datetime

from airflow import DAG
from airflow.models.param import Param
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from kubernetes.client import models as k8s


with DAG(
    dag_id="frascati_generation_simulation_workflow",
    start_date=datetime.fromisoformat("2026-05-11"),
    schedule=None,
    catchup=False,
    is_paused_upon_creation=False,
    params={
    "scene_name": Param(default='frascati', type='string', title='Scene Name', description='Scene id name.'),
    "target_lat": Param(default=41.808, type='number', title='Target Lat', description="Target's center latitude."),
    "target_lon": Param(default=12.681, type='number', title='Target Lon', description="Target's center longitude."),
    "target_size": Param(default=10.0, type='number', title='Target Size', description="Target's size in [km]."),
    "gmt_hour": Param(default=11.0, type='number', title='Gmt Hour', description='Hour of observation at target in GMT time.'),
    "spp": Param(default=8, type='integer', title='Spp', description='Number of Monte Carlo samples.'),
    "config_output_dir_generation": Param(default={'value': '/mnt/s2gos-output/gen_config', 'cid': None}, type='object', description='Generation configuration output directory.', nullable=True, properties={'value': {'type': 'string', 'title': 'Value', 'description': 'Full path URI'}, 'cid': {'title': 'Cid', 'description': 'Credential ID'}}, required=['value']),
    "scene_output_dir_generation": Param(default={'value': '/mnt/s2gos-output/gen_output', 'cid': None}, type='object', description='Scene description output directory.', nullable=True, properties={'value': {'type': 'string', 'title': 'Value', 'description': 'Full path URI'}, 'cid': {'title': 'Cid', 'description': 'Credential ID'}}, required=['value']),
    "config_output_dir_simulation": Param(default={'value': '/mnt/s2gos-output/sim_config', 'cid': None}, type='object', description='Simulation configuration output directory.', nullable=True, properties={'value': {'type': 'string', 'title': 'Value', 'description': 'Full path URI'}, 'cid': {'title': 'Cid', 'description': 'Credential ID'}}, required=['value']),
    "output_dir_simulation": Param(default={'value': '/mnt/s2gos-output/sim_output', 'cid': None}, type='object', description='Simulation output directory.', nullable=True, properties={'value': {'type': 'string', 'title': 'Value', 'description': 'Full path URI'}, 'cid': {'title': 'Cid', 'description': 'Credential ID'}}, required=['value'])
    },
) as dag:

    tasks = {}


    tasks["frascati_generation_simulation_workflow"] = KubernetesPodOperator(
        task_id="frascati_generation_simulation_workflow",
        image="quay.io/s2gos/s2gos-mono:0.0.5",
        cmds=["python", "/app/run_step.py"],
        arguments=[json.dumps({
            "func_module": "s2gos_apps.processes.frascati_workflow",
            "func_qualname": "frascati_generation_simulation_workflow",
            "inputs": {"scene_name": "{{ params.scene_name }}",
"target_lat": "{{ params.target_lat }}",
"target_lon": "{{ params.target_lon }}",
"target_size": "{{ params.target_size }}",
"gmt_hour": "{{ params.gmt_hour }}",
"spp": "{{ params.spp }}",
"config_output_dir_generation": "{{ params.config_output_dir_generation }}",
"scene_output_dir_generation": "{{ params.scene_output_dir_generation }}",
"config_output_dir_simulation": "{{ params.config_output_dir_simulation }}",
"output_dir_simulation": "{{ params.output_dir_simulation }}"},
            "output_keys": ['scene_name', 'target_lat', 'target_lon', 'target_size', 'gmt_hour', 'spp', 'config_path', 'scene_output_dir', 'config_output_dir_simulation', 'output_dir_simulation'],
        })],
        env_from=[k8s.V1EnvFromSource(secret_ref=k8s.V1SecretEnvSource(name='s2gos-credentials'))],
        container_resources=k8s.V1ResourceRequirements(requests={'cpu': '2', 'memory': '8Gi'}, limits={'cpu': '4', 'memory': '16Gi'}),
        volumes=[k8s.V1Volume(name='s2gos-output', persistent_volume_claim=k8s.V1PersistentVolumeClaimVolumeSource(claim_name='s2gos-output-pvc')), k8s.V1Volume(name='s2gos-settings', config_map=k8s.V1ConfigMapVolumeSource(name='s2gos-settings'))],
        volume_mounts=[k8s.V1VolumeMount(name='s2gos-output', mount_path='/mnt/output'), k8s.V1VolumeMount(name='s2gos-settings', mount_path='/opt/pixi/', sub_path='s2gos_settings.yaml')],
        do_xcom_push=True,
    )


    tasks["frascati-generation"] = KubernetesPodOperator(
        task_id="frascati-generation",
        image="quay.io/s2gos/s2gos-mono:0.0.5",
        cmds=["python", "/app/run_step.py"],
        arguments=[json.dumps({
            "func_module": "s2gos_apps.processes.frascati_workflow",
            "func_qualname": "frascati_generation",
            "inputs": {"config_path": "{{ ti.xcom_pull(task_ids='frascati_generation_simulation_workflow')['config_path'] }}"},
            "output_keys": ['scene_description_path'],
        })],
        env_from=[k8s.V1EnvFromSource(secret_ref=k8s.V1SecretEnvSource(name='s2gos-credentials'))],
        container_resources=k8s.V1ResourceRequirements(requests={'cpu': '2', 'memory': '8Gi'}, limits={'cpu': '4', 'memory': '16Gi'}),
        volumes=[k8s.V1Volume(name='s2gos-output', persistent_volume_claim=k8s.V1PersistentVolumeClaimVolumeSource(claim_name='s2gos-output-pvc')), k8s.V1Volume(name='s2gos-settings', config_map=k8s.V1ConfigMapVolumeSource(name='s2gos-settings'))],
        volume_mounts=[k8s.V1VolumeMount(name='s2gos-output', mount_path='/mnt/output'), k8s.V1VolumeMount(name='s2gos-settings', mount_path='/opt/pixi/', sub_path='s2gos_settings.yaml')],
        do_xcom_push=True,
    )


    tasks["frascati-simulation-config"] = KubernetesPodOperator(
        task_id="frascati-simulation-config",
        image="quay.io/s2gos/s2gos-mono:0.0.5",
        cmds=["python", "/app/run_step.py"],
        arguments=[json.dumps({
            "func_module": "s2gos_apps.processes.frascati_workflow",
            "func_qualname": "simulation_configs",
            "inputs": {"scene_name": "{{ ti.xcom_pull(task_ids='frascati_generation_simulation_workflow')['scene_name'] }}",
"target_lat": "{{ ti.xcom_pull(task_ids='frascati_generation_simulation_workflow')['target_lat'] }}",
"target_lon": "{{ ti.xcom_pull(task_ids='frascati_generation_simulation_workflow')['target_lon'] }}",
"target_size": "{{ ti.xcom_pull(task_ids='frascati_generation_simulation_workflow')['target_size'] }}",
"gmt_hour": "{{ ti.xcom_pull(task_ids='frascati_generation_simulation_workflow')['gmt_hour'] }}",
"spp": "{{ ti.xcom_pull(task_ids='frascati_generation_simulation_workflow')['spp'] }}",
"config_output_dir": "{{ ti.xcom_pull(task_ids='frascati_generation_simulation_workflow')['config_output_dir_simulation'] }}"},
            "output_keys": ['config_path'],
        })],
        env_from=[k8s.V1EnvFromSource(secret_ref=k8s.V1SecretEnvSource(name='s2gos-credentials'))],
        container_resources=k8s.V1ResourceRequirements(requests={'cpu': '2', 'memory': '8Gi'}, limits={'cpu': '4', 'memory': '16Gi'}),
        volumes=[k8s.V1Volume(name='s2gos-output', persistent_volume_claim=k8s.V1PersistentVolumeClaimVolumeSource(claim_name='s2gos-output-pvc')), k8s.V1Volume(name='s2gos-settings', config_map=k8s.V1ConfigMapVolumeSource(name='s2gos-settings'))],
        volume_mounts=[k8s.V1VolumeMount(name='s2gos-output', mount_path='/mnt/output'), k8s.V1VolumeMount(name='s2gos-settings', mount_path='/opt/pixi/', sub_path='s2gos_settings.yaml')],
        do_xcom_push=True,
    )


    tasks["frascati-simulation"] = KubernetesPodOperator(
        task_id="frascati-simulation",
        image="quay.io/s2gos/s2gos-mono:0.0.5",
        cmds=["python", "/app/run_step.py"],
        arguments=[json.dumps({
            "func_module": "s2gos_apps.processes.frascati_workflow",
            "func_qualname": "frascati_simulation",
            "inputs": {"scene_description_path": "{{ ti.xcom_pull(task_ids='frascati-generation')['scene_description_path'] }}",
"config_path": "{{ ti.xcom_pull(task_ids='frascati-simulation-config')['config_path'] }}",
"simulation_output_dir": "{{ ti.xcom_pull(task_ids='frascati_generation_simulation_workflow')['output_dir_simulation'] }}"},
            "output_keys": ['simulation_path'],
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
            "upstream_task_id": "frascati-simulation"
        },
        do_xcom_push=True
    )

    tasks["frascati_generation_simulation_workflow"] >> tasks["frascati-generation"]
    tasks["frascati_generation_simulation_workflow"] >> tasks["frascati-simulation-config"]
    tasks["frascati-generation"] >> tasks["frascati-simulation"]
    tasks["frascati-simulation-config"] >> tasks["frascati-simulation"]
    tasks["frascati_generation_simulation_workflow"] >> tasks["frascati-simulation"]
    tasks["frascati-simulation"] >> tasks["__procodile_final_step__"]

