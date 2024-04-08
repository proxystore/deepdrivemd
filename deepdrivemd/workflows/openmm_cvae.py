"""DeepDriveMD using OpenMM for simulation and a convolutional
variational autoencoder for adaptive control."""

import logging
import time
from argparse import ArgumentParser
from functools import partial, update_wrapper
from pathlib import Path
from queue import Queue
from typing import Any

from colmena.models import Result
from colmena.queue.python import PipeQueues
from colmena.task_server import ParslTaskServer
from colmena.thinker import agent, event_responder
from proxystore.store import register_store
from proxystore.store.base import Store
from proxystore.connectors.file import FileConnector

from proxystore.connectors.redis import RedisConnector
from proxystore.store.future import Future
from proxystore.stream.interface import StreamConsumer, StreamProducer
from proxystore.stream.shims.redis import RedisQueuePublisher, RedisQueueSubscriber

from deepdrivemd.api import (  # InferenceCountDoneCallback,
    DeepDriveMDSettings,
    DeepDriveMDWorkflow,
    SimulationCountDoneCallback,
    TimeoutDoneCallback,
)
from deepdrivemd.apps.cvae_inference import (
    CVAEInferenceInput,
    CVAEInferenceOutput,
    CVAEInferenceSettings,
)
from deepdrivemd.apps.cvae_train import (
    CVAETrainInput,
    CVAETrainOutput,
    CVAETrainSettings,
)
from deepdrivemd.apps.openmm_simulation import (
    MDSimulationInput,
    MDSimulationOutput,
    MDSimulationSettings,
)
from deepdrivemd.parsl import ComputeSettingsTypes


def run_simulation(
    input_data: MDSimulationInput, config: MDSimulationSettings
) -> MDSimulationOutput:
    from deepdrivemd.apps.openmm_simulation.app import MDSimulationApplication

    app = MDSimulationApplication(config)
    output_data = app.run(input_data)
    return output_data


def run_train(input_data: CVAETrainInput, config: CVAETrainSettings) -> CVAETrainOutput:
    from deepdrivemd.apps.cvae_train.app import CVAETrainApplication

    app = CVAETrainApplication(config)
    output_data = app.run(input_data)
    return output_data


def run_inference(
    redis_host: str,
    redis_port: int,
    stop_inference: Future[bool],
    config: CVAEInferenceSettings,
) -> None:
    from deepdrivemd.apps.cvae_inference.app import CVAEInferenceApplication

    app = CVAEInferenceApplication(
        config,
        redis_host=redis_host,
        redis_port=redis_port,
        stop_inference=stop_inference,
    )
    app.run()


class DeepDriveMD_OpenMM_CVAE(DeepDriveMDWorkflow):
    def __init__(
        self,
        simulations_per_train: int,
        simulations_per_inference: int,
        redis_host: str,
        redis_port: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        # Make sure there has been at least one training task complete before running inference
        self.model_weights_available: bool = False

        # For batching training inputs
        self.simulations_per_train = simulations_per_train
        self.train_input = CVAETrainInput(contact_map_paths=[], rmsd_paths=[])

        # For batching inference inputs
        self.simulations_per_inference = simulations_per_inference
        self.inference_input = CVAEInferenceInput(
            contact_map_paths=[], rmsd_paths=[], model_weight_path=Path()
        )

        # Communicate results between agents
        self.simulation_input_queue: Queue[MDSimulationInput] = Queue()

        self.redis_host = redis_host
        self.redis_port = redis_port
        self.store = Store(
            "redis-store",
            RedisConnector(redis_host, redis_port),
        )

        # Inference result consumer
        self.inference_batch_index = 0
        self.stop_inference: Future[bool] | None = None
        publisher = RedisQueuePublisher(self.redis_host, self.redis_port)
        subscriber = RedisQueueSubscriber(
            self.redis_host,
            self.redis_port,
            topic="inference-output",
        )
        # Clear current queues
        publisher._redis_client.delete("inference-input", "inference-output")
        self.inference_output_consumer = StreamConsumer[CVAEInferenceOutput](
            subscriber,
        )
        self.inference_input_producer = StreamProducer[CVAEInferenceInput](
            publisher,
            {"inference-input": self.store, "inference-output": self.store},
        )

        # Signal we can start inference worker now.
        self.run_inference.set()

    def simulate(self) -> None:
        """Select a method to start another simulation. If AI inference
        is currently adding new restart points to the queue, we block
        until it has finished so we can use the latest information.

        In the first iteration, we simply cycle around the input directories.
        """
        with self.simulation_govenor:
            if not self.simulation_input_queue.empty():
                # If the AI inference has selected restart points, use those
                inputs = self.simulation_input_queue.get()
            else:
                # Otherwise, start an initial simulation
                inputs = MDSimulationInput(sim_dir=next(self.simulation_input_dirs))

        self.submit_task("simulation", inputs)

    def train(self) -> None:
        self.submit_task("train", self.train_input)
        # self.train_input.clear()  # Clear batched data

    def inference(self) -> None:
        # Inference must wait for a trained model to be available
        while not self.model_weights_available:
            time.sleep(1)

        if self.stop_inference is None:
            # First inference batch so we need to submit the task
            self.stop_inference: Future[bool] = store.future()
            self.submit_task(
                "inference",
                self.redis_host,
                self.redis_port,
                self.stop_inference,
            )

        self.logger.info(
            "Sending inference input to 'inference-input' stream "
            f"(batch: {self.inference_batch_index})"
        )
        self.inference_input_producer.send(
            "inference-input",
            self.inference_input,
            metadata={"index": self.inference_batch_index},
        )
        self.inference_batch_index += 1
        # self.inference_input.clear()  # Clear batched data

    def handle_simulation_output(self, output: MDSimulationOutput) -> None:
        # Collect simulation results
        self.train_input.append(output.contact_map_path, output.rmsd_path)
        self.inference_input.append(output.contact_map_path, output.rmsd_path)
        # Since we are not clearing the train/inference inputs, the length will be the same
        num_sims = len(self.train_input)

        if num_sims and (num_sims % self.simulations_per_train == 0):
            self.run_training.set()

        if num_sims and (num_sims % self.simulations_per_inference == 0):
            self.run_inference.set()

    def handle_train_output(self, output: CVAETrainOutput) -> None:
        self.inference_input.model_weight_path = output.model_weight_path
        self.model_weights_available = True
        self.logger.info(f"Updated model_weight_path to: {output.model_weight_path}")

    def handle_inference_output(self, output: CVAEInferenceOutput) -> None:
        # Add restart points to simulation input queue while holding the lock
        # so that the simulations see the latest information. Note that
        # the output restart values should be sorted such that the first
        # element in sim_dirs and sim_frames is the leading restart point.
        with self.simulation_govenor:
            # First empty the queue of old outliers
            self.simulation_input_queue.queue.clear()
            # Then fill it back up with new outliers
            for sim_dir, sim_frame in zip(output.sim_dirs, output.sim_frames):
                self.simulation_input_queue.put(
                    MDSimulationInput(sim_dir=sim_dir, sim_frame=sim_frame)
                )

        self.logger.info(
            f"processed inference result and added {len(output.sim_dirs)} "
            "new restart points to the simulation_input_queue."
        )

    @agent
    def stop_inference_output_stream(self) -> None:
        while not self.done.is_set():
            time.sleep(1)

        self.stop_inference.set_result(True)
        # This will cause the inference_output_consumer to raise
        # a StopIteration so handle_inference_output_stream breaks out
        # of it's loop.
        self.inference_input_producer.close_topics("inference-output")
        # Quick hack to allow us to reuse topic
        self.inference_input_producer._buffer["inference-input"].closed = False
        self.logger.info("Closed 'inference-output' stream")


class ExperimentSettings(DeepDriveMDSettings):
    """Provide a YAML interface to configure the experiment."""

    simulation_settings: MDSimulationSettings
    train_settings: CVAETrainSettings
    inference_settings: CVAEInferenceSettings
    compute_settings: ComputeSettingsTypes


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument(
        "-t", "--test", action="store_true", help="Test Mock Application"
    )
    parser.add_argument("--redis-host", help="Redis host.")
    parser.add_argument("--redis-port", type=int, help="Redis port.")
    args = parser.parse_args()
    cfg = ExperimentSettings.from_yaml(args.config)
    cfg.dump_yaml(cfg.run_dir / "params.yaml")
    cfg.configure_logging()

    # Make the proxy store
    connector = FileConnector(store_dir=str(cfg.run_dir / "proxy-store"))
    store = Store(name="file", connector=connector)
    register_store(store)

    # Make the queues
    queues = PipeQueues(
        serialization_method="pickle",
        topics=["simulation", "train", "inference"],
        proxystore_name="file",
        proxystore_threshold=10000,
    )

    # Define the parsl configuration (this can be done using the config_factory
    # for common use cases or by defining your own configuration.)
    parsl_config = cfg.compute_settings.config_factory(cfg.run_dir / "run-info")

    # Assign constant settings to each task function
    my_run_simulation = partial(run_simulation, config=cfg.simulation_settings)
    my_run_train = partial(run_train, config=cfg.train_settings)
    my_run_inference = partial(run_inference, config=cfg.inference_settings)
    update_wrapper(my_run_simulation, run_simulation)
    update_wrapper(my_run_train, run_train)
    update_wrapper(my_run_inference, run_inference)

    doer = ParslTaskServer(
        [my_run_simulation, my_run_train, my_run_inference], queues, parsl_config
    )

    thinker = DeepDriveMD_OpenMM_CVAE(
        queue=queues,
        result_dir=cfg.run_dir / "result",
        simulation_input_dir=cfg.simulation_input_dir,
        num_workers=cfg.num_workers,
        simulations_per_train=cfg.simulations_per_train,
        simulations_per_inference=cfg.simulations_per_inference,
        done_callbacks=[
            # InferenceCountDoneCallback(2),  # Testing
            SimulationCountDoneCallback(cfg.num_total_simulations),
            TimeoutDoneCallback(cfg.duration_sec),
        ],
        redis_host=args.redis_host,
        redis_port=args.redis_port,
    )
    logging.info("Created the task server and task generator")

    try:
        # Launch the servers
        doer.start()
        thinker.start()
        logging.info("Launched the servers")

        # Wait for the task generator to complete
        thinker.join()
        logging.info("Task generator has completed")
    finally:
        queues.send_kill_signal()

    # Wait for the task server to complete
    doer.join()

    # Clean up proxy store
    store.close()
