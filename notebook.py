# %% [markdown]
# # Proyek MLOps : Depression Students
# - **Nama:** Abid Juliant Indraswara
# - **Email:** abidjuliant@gmail.com
# - **ID Dicoding:** abidindraswara
# 
# Fokus pada proyek ini adalah untuk membuat pipeline machine learning dengan topik machine learning yaitu sentimen berita mengenai siswa yang mengalami depresi. Dataset yang digunakan adalah depression-student-dataset yang berasal dari Kaggle.
# 
# Dataset : https://www.kaggle.com/datasets/ikynahidwin/depression-student-dataset

# %% [markdown]
# ## Install & Import Libray

# %%
# Import Library Umum
import os, shutil
from shutil import copyfile
import zipfile
import pathlib
from pathlib import Path
import numpy as np
import pandas as pd

# %%
# Import Library untuk Machine Learning dan Pipeline
import tensorflow as tf
import tensorflow_model_analysis as tfma
from sklearn.preprocessing import LabelEncoder
from tfx.components import (
    CsvExampleGen,
    StatisticsGen,
    SchemaGen,
    ExampleValidator,
    Transform,
    Trainer,
    Evaluator,
    Pusher,
    Tuner
)
from tfx.proto import example_gen_pb2, trainer_pb2, pusher_pb2
from tfx.types import Channel
from tfx.dsl.components.common.resolver import Resolver
from tfx.orchestration.experimental.interactive.interactive_context import InteractiveContext
from tfx.types.standard_artifacts import Model, ModelBlessing
from tfx.dsl.input_resolution.strategies.latest_blessed_model_strategy import (
    LatestBlessedModelStrategy)

# %% [markdown]
# ## Data Loading & Cek Dataset

# %%
# Dataset Depression Students
depression_df = pd.read_csv("./depression_student/data/data.csv")
depression_df

# %% [markdown]
# ### Cek Nilai NaN Dataset

# %%
# Cek NaN Dataset
print('Cek NaN Dataset Depression Student')
print(depression_df.isna().sum())

# %% [markdown]
# ### Cek Info Dataset

# %%
# Cek Info Dataset
depression_df.info()

# %% [markdown]
# ### Cek Fitur Kategori

# %%
# Cek Kategori untuk Kolom dengan Tipe Data object
category_per_columns = depression_df.select_dtypes(include=['object']).columns

# %%
# Cek per Parameter atau kolom
for column in category_per_columns:
    category_value = depression_df[column].unique()
    print(f"{column}: \n{category_value}", '\n')

# %% [markdown]
# ## Encoding Label

# %%
# Inisialisasi LabelEncoder
label_encoder = LabelEncoder()

# Mengonversi kolom 'Depression' menjadi label numerik
depression_df['Depression'] = label_encoder.fit_transform(depression_df['Depression'])

# Tampilkan hasil label encoding
depression_df[['Depression']]

# %% [markdown]
# ### Convert Dataset File

# %%
# Convert dataset to csv file
depression_df.to_csv("./data/data.csv", index=False)

# %% [markdown]
# ## Running Pipeline Machine Learning

# %% [markdown]
# ### Import Library TFX

# %%
import os
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner
from modules.pipeline import init_local_pipeline
from modules.depression_components import init_components

# %% [markdown]
# ### Init Variabel Pipeline

# %%
PIPELINE_NAME = 'abidindraswara-pipeline'

DATA_ROOT = 'data'
TRANSFORM_MODULE_FILE = 'modules/depression_transform.py'
TRAINER_MODULE_FILE = 'modules/depression_trainer.py'

OUTPUT_BASE = 'output'
serving_model_dir = os.path.join(OUTPUT_BASE, 'serving_model')
pipeline_root = os.path.join(OUTPUT_BASE, PIPELINE_NAME)
metadata_path = os.path.join(pipeline_root, 'metadata.sqlite')

# %% [markdown]
# ### Running Pipeline secara Lokal

# %%
components = init_components(
    data_dir=DATA_ROOT,
    transform_module=TRANSFORM_MODULE_FILE,
    training_module=TRAINER_MODULE_FILE,
    training_steps=5000,
    eval_steps=1000,
    serving_model_dir=serving_model_dir
)

pipeline = init_local_pipeline(components, pipeline_root)
BeamDagRunner().run(pipeline)


