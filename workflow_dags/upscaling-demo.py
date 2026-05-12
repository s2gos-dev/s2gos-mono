# WARNING - THIS IS GENERATED CODE
#   Generator: Eozilla Appligator v0.1.0
#        Date: 2026-05-12T11:18:21.695062

import json
from datetime import datetime

from airflow import DAG
from airflow.models.param import Param
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from kubernetes.client import models as k8s


with DAG(
    dag_id="upscaling-demo",
    start_date=datetime.fromisoformat("2026-05-11"),
    schedule=None,
    catchup=False,
    is_paused_upon_creation=False,
    params={
    "scene_name": Param(default='pnp', type='string', title='Locations', description='Scene name.', enum=['pnp', 'pisa']),
    "month": Param(default='June', type='string', title='Month', description='Month.', enum=['June']),
    "day": Param(default=1, type='integer', title='Day', description='Day of the month.', minimum=1, maximum=30),
    "hour": Param(type='number', title='Hour', description='Time of day. For satelites, the closest overpass time will be used'),
    "include_TLS": Param(default=False, type='boolean', title='Include Tls', description='Include Telestrial Laser Scanned data.'),
    "observation": Param(title='Observation', description='Observations.', anyOf=[{'type': 'object', 'title': 'SatelliteObservation', 'properties': {'satellite_instrument': {'type': 'string', 'title': 'SatelliteInstrument', 'description': 'Satellite Instrument', 'enum': ['CHIME', 'MSI'], 'default': 'CHIME'}, 'spp': {'type': 'integer', 'title': 'Spp', 'description': 'Sample Per Pixel', 'default': 8}, 'orthorectified': {'type': 'boolean', 'title': 'Orthorectified', 'description': 'Specifies whether the simulation is done in sensor space or target space.', 'default': True}, 'psf': {'type': 'boolean', 'title': 'Psf', 'description': 'Point spread function.', 'default': False}, 'srf': {'type': 'boolean', 'title': 'Srf', 'description': 'Spectral response function.', 'default': True}, 'radiometric_noise': {'type': 'number', 'title': 'Radiometric Noise', 'description': '', 'default': 0.0}}}, {'type': 'object', 'title': 'GroundObservation', 'properties': {'observation_type': {'type': 'string', 'title': 'GroundObservationType', 'description': 'Ground observation type', 'enum': ['Hypstar HCRF', 'camera'], 'default': 'Hypstar HCRF'}, 'spp': {'type': 'integer', 'title': 'Spp', 'description': 'Sample Per Pixel', 'default': 8}}}, {'type': 'object', 'title': 'SurfaceL2', 'properties': {'L2_product': {'type': 'string', 'title': 'SurfaceL2Type', 'description': 'L2 Product', 'enum': ['HDRF'], 'default': 'HDRF'}, 'footprint': {'type': 'number', 'title': 'Footprint', 'description': 'Pixel footprint resolution in meters', 'default': 30.0}, 'satellite': {'type': 'string', 'title': 'SatelliteInstrument', 'description': 'If specified, informs the pixel footprint of the L2 product. Takes precedence over `footprint`.', 'enum': ['CHIME', 'MSI'], 'nullable': True}, 'spp': {'type': 'integer', 'title': 'Spp', 'description': 'Sample Per Pixel', 'default': 8}}}]),
    "config_output_dir": Param(type='object', description='Generation configuration output directory.', nullable=True, properties={'value': {'type': 'string', 'title': 'Value', 'description': 'Full path URI'}, 'cid': {'title': 'Cid', 'description': 'Credential ID'}}, required=['value']),
    "scene_output_dir": Param(type='object', description='Scene description output directiory.', nullable=True, properties={'value': {'type': 'string', 'title': 'Value', 'description': 'Full path URI'}, 'cid': {'title': 'Cid', 'description': 'Credential ID'}}, required=['value'])
    },
) as dag:

    tasks = {}


    tasks["upscaling-demo"] = KubernetesPodOperator(
        task_id="upscaling-demo",
        image="quay.io/s2gos/s2gos-mono:0.0.5",
        cmds=["python", "/app/run_step.py"],
        arguments=[json.dumps({
            "func_module": "s2gos_apps.processes.upscaling",
            "func_qualname": "upscaling",
            "inputs": {"scene_name": "{{ params.scene_name }}",
"month": "{{ params.month }}",
"day": "{{ params.day }}",
"hour": "{{ params.hour }}",
"include_TLS": "{{ params.include_TLS }}",
"observation": "{{ params.observation }}",
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
            "upstream_task_id": "upscaling-demo"
        },
        do_xcom_push=True
    )

    tasks["upscaling-demo"] >> tasks["__procodile_final_step__"]

