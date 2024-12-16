"""
    Author      : Abid Indraswara
    Date Create : 13 December 2024
    Modul Pipeline
    Fungsi :
    - Modul untuk menjalankan keseluruhan komponen pipeline ML.
"""

# Import Library Umum
import os
from typing import Text

# Import Library TFX
from absl import logging
from tfx.orchestration import metadata, pipeline
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner

# Nama Pipeline
PIPELINE_NAME = 'abidindraswara-pipeline'

# Path Dataset
DATA_ROOT = 'data'
# Path Modul Transform
TRANSFORM_MODULE_FILE = 'modules/depression_transform.py'
# Path Modul Trainer
TRAINER_MODULE_FILE = 'modules/depression_trainer.py'
# Path Folder Outpue
OUTPUT_BASE = 'output'
# Path Folder Serving Model
serving_model_dir = os.path.join(OUTPUT_BASE, 'serving_model')
pipeline_root = os.path.join(OUTPUT_BASE, PIPELINE_NAME)
metadata_path = os.path.join(pipeline_root, 'metadata.sqlite')

# Menjalankan Fungsi Inisiasi Pipeline secara Lokal
def init_local_pipeline(
    components, pipeline_root: Text
) -> pipeline.Pipeline:
    """
    Initialize a local TFX pipeline.

    Args:
        components: A dictionary of TFX components to be included in the pipeline.
        pipeline_root: Root directory for pipeline output artifacts.

    Returns:
        A TFX pipeline.
    """
    logging.info(f'Pipeline root set to: {pipeline_root}')
    logging.info(f'Metadata path set to: {metadata_path}')

    beam_args = [
        '--direct_running_mode=multi_processing',
        '--direct_num_workers=0' 
    ]

    return pipeline.Pipeline(
        pipeline_name=PIPELINE_NAME,
        pipeline_root=pipeline_root,
        components=components,
        enable_cache=True,
        metadata_connection_config=metadata.sqlite_metadata_connection_config(
            metadata_path
        ),
        beam_pipeline_args=beam_args
    )


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)

    from modules.depression_components import init_components
    
    components = init_components(
        DATA_ROOT,
        training_module=TRAINER_MODULE_FILE,
        transform_module=TRANSFORM_MODULE_FILE,
        training_steps=5000,
        eval_steps=1000,
        serving_model_dir=serving_model_dir,
    )

    pipeline = init_local_pipeline(components, pipeline_root)
    BeamDagRunner().run(pipeline=pipeline)
