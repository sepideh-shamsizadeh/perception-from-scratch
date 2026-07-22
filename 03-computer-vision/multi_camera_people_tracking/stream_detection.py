import argparse
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
from ultralytics import YOLO

from detection_tracking import (
    IMAGE_DIR,
    PROJECT_DIR,
    run_detection_on_frame,
)


@dataclass
class StreamFrame:
    stream_name: str
    frame_name: str
    frame: object
    captured_at: float
    sequence_number: int


@dataclass
class StreamStatistics:
    produced: int = 0
    processed: int = 0
    dropped: int = 0
    failed_reads: int = 0
    total_queue_delay: float = 0.0
    total_processing_time: float = 0.0
    maximum_queue_delay: float = 0.0


class LatestFrameQueue:
    """
    A bounded queue that drops the oldest frame when full.

    This prevents an increasingly large latency backlog.
    """

    def __init__(self, maximum_size: int = 1):
        if maximum_size < 1:
            raise ValueError("maximum_size must be at least 1.")

        self._queue = queue.Queue(maxsize=maximum_size)
        self._lock = threading.Lock()
        self.dropped_frames = 0

    def put_latest(self, item: StreamFrame) -> None:
        with self._lock:
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                    self._queue.task_done()
                    self.dropped_frames += 1
                except queue.Empty:
                    pass

            self._queue.put_nowait(item)

    def get(self, timeout: Optional[float] = None) -> StreamFrame:
        return self._queue.get(timeout=timeout)

    def task_done(self) -> None:
        self._queue.task_done()

    def empty(self) -> bool:
        return self._queue.empty()


class WildtrackImageStream(threading.Thread):
    """
    Simulates a live camera stream using sequential Wildtrack images.

    Images are emitted according to target_fps rather than processed
    as quickly as possible.
    """

    def __init__(
        self,
        camera_name: str,
        output_queue: LatestFrameQueue,
        target_fps: float = 10.0,
        repeat: bool = False,
        maximum_frames: Optional[int] = None,
    ):
        super().__init__(daemon=True)

        if target_fps <= 0:
            raise ValueError("target_fps must be positive.")

        self.camera_name = camera_name
        self.output_queue = output_queue
        self.target_fps = target_fps
        self.repeat = repeat
        self.maximum_frames = maximum_frames

        self.stop_event = threading.Event()
        self.finished_event = threading.Event()
        self.statistics = StreamStatistics()

    def stop(self) -> None:
        self.stop_event.set()

    def _get_frame_paths(self) -> list[Path]:
        camera_directory = IMAGE_DIR / self.camera_name

        if not camera_directory.exists():
            raise FileNotFoundError(
                f"Camera directory does not exist: {camera_directory}"
            )

        frame_paths = sorted(
            camera_directory.glob("*.png"),
            key=lambda path: int(path.stem),
        )

        if self.maximum_frames is not None:
            frame_paths = frame_paths[: self.maximum_frames]

        if not frame_paths:
            raise RuntimeError(
                f"No PNG frames found in {camera_directory}"
            )

        return frame_paths

    def run(self) -> None:
        try:
            frame_paths = self._get_frame_paths()
            frame_period = 1.0 / self.target_fps
            sequence_number = 0

            while not self.stop_event.is_set():
                stream_start = time.perf_counter()

                for frame_path in frame_paths:
                    if self.stop_event.is_set():
                        break

                    target_time = (
                        stream_start
                        + sequence_number * frame_period
                    )

                    sleep_time = target_time - time.perf_counter()

                    if sleep_time > 0:
                        time.sleep(sleep_time)

                    captured_at = time.perf_counter()
                    frame = cv2.imread(str(frame_path))

                    if frame is None:
                        self.statistics.failed_reads += 1
                        continue

                    stream_frame = StreamFrame(
                        stream_name=self.camera_name,
                        frame_name=frame_path.stem,
                        frame=frame,
                        captured_at=captured_at,
                        sequence_number=sequence_number,
                    )

                    dropped_before = (
                        self.output_queue.dropped_frames
                    )

                    self.output_queue.put_latest(stream_frame)

                    dropped_after = (
                        self.output_queue.dropped_frames
                    )

                    self.statistics.dropped += (
                        dropped_after - dropped_before
                    )
                    self.statistics.produced += 1
                    sequence_number += 1

                if not self.repeat:
                    break

        finally:
            self.finished_event.set()


def draw_predictions(frame, predictions):
    output = frame.copy()

    for prediction in predictions:
        x1, y1, x2, y2 = (
            int(value)
            for value in prediction["box"]
        )

        confidence = prediction["confidence"]

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            2,
        )

        cv2.putText(
            output,
            f"person {confidence:.2f}",
            (x1, max(y1 - 5, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    return output


def warm_up_streaming_path(model, camera_name, iterations=10):
    camera_dir = IMAGE_DIR / camera_name

    frame_path = sorted(
        camera_dir.glob("*.png"),
        key=lambda path: int(path.stem),
    )[0]

    frame = cv2.imread(str(frame_path))

    if frame is None:
        raise RuntimeError(f"Could not read {frame_path}")

    print("Warming up exact streaming path...")

    for _ in range(iterations):
        run_detection_on_frame(
            model=model,
            frame=frame,
        )

    print("Streaming path warm-up complete.")


def run_stream_experiment(
    model_path: Path,
    camera_name: str,
    stream_fps: float,
    queue_size: int,
    maximum_frames: int,
    processing_delay_ms: float = 0.0,
    save_output: bool = False,
):
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model does not exist: {model_path}"
        )

    if model_path.suffix == ".engine":
        model = YOLO(str(model_path), task="detect")
    else:
        model = YOLO(str(model_path))

    warm_up_streaming_path(
        model=model,
        camera_name=camera_name,
        iterations=10,
    )

    frame_queue = LatestFrameQueue(
        maximum_size=queue_size
    )

    producer = WildtrackImageStream(
        camera_name=camera_name,
        output_queue=frame_queue,
        target_fps=stream_fps,
        repeat=False,
        maximum_frames=maximum_frames,
    )

    statistics = StreamStatistics()

    output_directory = (
        PROJECT_DIR
        / "stream_output"
        / camera_name
    )

    if save_output:
        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    producer.start()

    experiment_started = time.perf_counter()

    try:
        while True:
            producer_finished = (
                producer.finished_event.is_set()
            )

            if producer_finished and frame_queue.empty():
                break

            try:
                stream_frame = frame_queue.get(
                    timeout=0.5
                )
            except queue.Empty:
                continue

            inference_started = time.perf_counter()

            queue_delay = (
                inference_started
                - stream_frame.captured_at
            )

            predictions, processing_time, model_speed = (
                run_detection_on_frame(
                    model=model,
                    frame=stream_frame.frame,
                )
            )

            # Optional artificial delay to simulate heavier
            # downstream tracking or server processing.
            if processing_delay_ms > 0:
                time.sleep(processing_delay_ms / 1000.0)

            statistics.processed += 1
            statistics.total_queue_delay += queue_delay
            statistics.total_processing_time += (
                processing_time
            )
            statistics.maximum_queue_delay = max(
                statistics.maximum_queue_delay,
                queue_delay,
            )

            current_end_to_end_latency = (
                time.perf_counter()
                - stream_frame.captured_at
            )

            print(
                f"stream={stream_frame.stream_name} "
                f"frame={stream_frame.frame_name} "
                f"sequence={stream_frame.sequence_number} "
                f"detections={len(predictions)} "
                f"queue_delay={queue_delay * 1000:.1f}ms "
                f"inference={model_speed['inference']:.1f}ms "
                f"processing={processing_time * 1000:.1f}ms "
                f"capture_to_output="
                f"{current_end_to_end_latency * 1000:.1f}ms"
            )

            if save_output:
                output_frame = draw_predictions(
                    stream_frame.frame,
                    predictions,
                )

                output_path = (
                    output_directory
                    / f"{stream_frame.frame_name}.jpg"
                )

                cv2.imwrite(
                    str(output_path),
                    output_frame,
                )

            frame_queue.task_done()

    finally:
        producer.stop()
        producer.join(timeout=2.0)

    experiment_duration = (
        time.perf_counter()
        - experiment_started
    )

    statistics.produced = producer.statistics.produced
    statistics.dropped = frame_queue.dropped_frames
    statistics.failed_reads = (
        producer.statistics.failed_reads
    )

    mean_queue_delay_ms = (
        statistics.total_queue_delay
        / statistics.processed
        * 1000
        if statistics.processed
        else 0.0
    )

    mean_processing_ms = (
        statistics.total_processing_time
        / statistics.processed
        * 1000
        if statistics.processed
        else 0.0
    )

    effective_processing_fps = (
        statistics.processed / experiment_duration
        if experiment_duration > 0
        else 0.0
    )

    drop_rate = (
        statistics.dropped / statistics.produced
        if statistics.produced
        else 0.0
    )

    print("\n=== Streaming summary ===")
    print(f"Model: {model_path.name}")
    print(f"Camera: {camera_name}")
    print(f"Configured stream FPS: {stream_fps:.2f}")
    print(f"Queue capacity: {queue_size}")
    print(f"Produced frames: {statistics.produced}")
    print(f"Processed frames: {statistics.processed}")
    print(f"Dropped frames: {statistics.dropped}")
    print(f"Failed reads: {statistics.failed_reads}")
    print(f"Drop rate: {drop_rate:.2%}")
    print(
        f"Mean queue delay: "
        f"{mean_queue_delay_ms:.2f} ms"
    )
    print(
        f"Maximum queue delay: "
        f"{statistics.maximum_queue_delay * 1000:.2f} ms"
    )
    print(
        f"Mean processing time: "
        f"{mean_processing_ms:.2f} ms"
    )
    print(
        f"Effective processed FPS: "
        f"{effective_processing_fps:.2f}"
    )


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Simulate a live Wildtrack camera stream and "
            "run YOLO detection with bounded buffering."
        )
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_DIR / "yolo11n_fp16.engine",
    )

    parser.add_argument(
        "--camera",
        type=str,
        default="C1",
    )

    parser.add_argument(
        "--stream-fps",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--queue-size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--frames",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--processing-delay-ms",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--save-output",
        action="store_true",
    )

    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()

    run_stream_experiment(
        model_path=arguments.model,
        camera_name=arguments.camera,
        stream_fps=arguments.stream_fps,
        queue_size=arguments.queue_size,
        maximum_frames=arguments.frames,
        processing_delay_ms=(
            arguments.processing_delay_ms
        ),
        save_output=arguments.save_output,
    )