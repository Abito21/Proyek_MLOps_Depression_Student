"""
    Author      : Abid Indraswara
    Date Create : 13 December 2024
    Inisiasi Komponen Pipeline TFX
    Fungsi :
    - Inisiasi segala komponen TFX sebelum dijalankan berisi
      komponen ingestion, validator, schema, transform, tuner,
      trainer, resolver, evaluator dan pusher.
"""

# Import Library Umum
import os

# Import Library Tensorflow
import tensorflow as tf
import tensorflow_model_analysis as tfma

# Import Library TFX
from tfx.components import (
    CsvExampleGen,
    StatisticsGen,
    SchemaGen,
    ExampleValidator,
    Transform,
    Trainer,
    Evaluator,
    Pusher
)
from tfx.proto import example_gen_pb2, trainer_pb2, pusher_pb2
from tfx.types import Channel
from tfx.dsl.components.common.resolver import Resolver
from tfx.types.standard_artifacts import Model, ModelBlessing
from tfx.dsl.input_resolution.strategies.latest_blessed_model_strategy import (
    LatestBlessedModelStrategy)

# Fungsi Inisisasi Komponen
def init_components(
    data_dir,
    transform_module,
    training_module,
    training_steps,
    eval_steps,
    serving_model_dir,
):
    """Inisiasi Komponen Pipeline TFX

    Args:
        data_dir (str): a path to the data
        transform_module (str): a path to the transform_module
        training_module (str): a path to the transform_module
        training_steps (int): number of training steps
        eval_steps (int): number of eval steps
        serving_model_dir (str): a path to the serving model directory

    Returns:
        TFX components
    """

    # Komponen Data Ingestion
    output = example_gen_pb2.Output(
        split_config = example_gen_pb2.SplitConfig(splits=[
            example_gen_pb2.SplitConfig.Split(name="train", hash_buckets=8),
            example_gen_pb2.SplitConfig.Split(name="eval", hash_buckets=2)
        ])
    )

    example_gen = CsvExampleGen(
        input_base=data_dir,
        output_config=output
    )

    # Komponen Data Validation
    statistics_gen = StatisticsGen(
        examples=example_gen.outputs["examples"]
    )

    # Komponen Data Schema
    schema_gen = SchemaGen(
        statistics=statistics_gen.outputs["statistics"]
    )

    # Komponen Identifikasi Anomali Data
    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs['statistics'],
        schema=schema_gen.outputs['schema']
    )

    # Komponen Transform Data
    transform  = Transform(
        examples=example_gen.outputs['examples'],
        schema= schema_gen.outputs['schema'],
        module_file=os.path.abspath(transform_module)
    )

    # Komponen Training Model
    trainer  = Trainer(
        module_file=os.path.abspath(training_module),
        examples = transform.outputs['transformed_examples'],
        transform_graph=transform.outputs['transform_graph'],
        schema=schema_gen.outputs['schema'],
        train_args=trainer_pb2.TrainArgs(
            splits=['train'],
            num_steps=training_steps),
        eval_args=trainer_pb2.EvalArgs(
            splits=['eval'],
            num_steps=eval_steps)
    )

    # Komponen Resolver
    model_resolver = Resolver(
        strategy_class= LatestBlessedModelStrategy,
        model = Channel(type=Model),
        model_blessing = Channel(type=ModelBlessing)
    ).with_id('Latest_blessed_model_resolver')

    # Komponen Evaluator
    # Setup Slicing
    slicing_specs=[
        tfma.SlicingSpec()
    ]

    # Setup Metrics
    metrics_specs = [
        tfma.MetricsSpec(metrics=[
                tfma.MetricConfig(class_name='AUC'),
                tfma.MetricConfig(class_name="Precision"),
                tfma.MetricConfig(class_name="Recall"),
                tfma.MetricConfig(class_name="ExampleCount"),
                tfma.MetricConfig(class_name='BinaryAccuracy',
                    threshold=tfma.MetricThreshold(
                        value_threshold=tfma.GenericValueThreshold(
                            lower_bound={'value':0.5}),
                        change_threshold=tfma.GenericChangeThreshold(
                            direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                            absolute={'value':0.0001})
                        )
                )
            ])
    ]

    # Setup Eval
    eval_config = tfma.EvalConfig(
        model_specs=[tfma.ModelSpec(label_key='Depression')],
        slicing_specs=slicing_specs,
        metrics_specs=metrics_specs
    )

    # Running Evaluator
    evaluator = Evaluator(
        examples=example_gen.outputs['examples'],
        model=trainer.outputs['model'],
        baseline_model=model_resolver.outputs['model'],
        eval_config=eval_config)

    # Komponen Pusher
    pusher = Pusher(
        model=trainer.outputs["model"],
        model_blessing=evaluator.outputs["blessing"],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=serving_model_dir
            )
        ),
    )

    # Daftar Komponen
    components = (
        example_gen,
        statistics_gen,
        schema_gen,
        example_validator,
        transform,
        trainer,
        model_resolver,
        evaluator,
        pusher
    )

    return components