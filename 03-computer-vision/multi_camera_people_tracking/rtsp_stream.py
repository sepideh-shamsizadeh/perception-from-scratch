import threading
import time

import cv2

from stream_detection import StreamFrame, StreamStatistics


class RTSPStream(threading.Thread):
    """
    Reads frames from a live RTSP camera into a LatestFrameQueue.

    Counterpart to WildtrackImageStream, which simulates a stream from
    the static dataset instead of a real camera.
    """

    def __init__(
        self,
        stream_name,
        rtsp_url,
        output_queue,
    ):
        super().__init__(daemon=True)

        self.stream_name = stream_name
        self.rtsp_url = rtsp_url
        self.output_queue = output_queue
        self.stop_event = threading.Event()
        self.finished_event = threading.Event()
        self.statistics = StreamStatistics()

    def stop(self):
        self.stop_event.set()

    def run(self):
        capture = cv2.VideoCapture(self.rtsp_url)

        # Backend support for this property can vary.
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not capture.isOpened():
            self.finished_event.set()
            raise RuntimeError(
                f"Could not open RTSP stream: "
                f"{self.rtsp_url}"
            )

        sequence_number = 0

        try:
            while not self.stop_event.is_set():
                success, frame = capture.read()
                captured_at = time.perf_counter()

                if not success:
                    self.statistics.failed_reads += 1
                    time.sleep(0.05)
                    continue

                item = StreamFrame(
                    stream_name=self.stream_name,
                    frame_name=str(sequence_number),
                    frame=frame,
                    captured_at=captured_at,
                    sequence_number=sequence_number,
                )

                self.output_queue.put_latest(item)

                self.statistics.produced += 1
                sequence_number += 1

        finally:
            capture.release()
            self.finished_event.set()
