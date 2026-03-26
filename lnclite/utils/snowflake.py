import hashlib
import os
import socket
import threading
import time
from typing import Optional

_global_generator: Optional["Snowflake"] = None


class Snowflake:
    def __init__(self, worker_id: int):
        self.worker_id = worker_id
        self.sequence = 0
        self.last_timestamp = -1

        self.twepoch = 1704067200000  # 2024-01-01

        # Define the bit lengths of each part
        self.worker_id_bits = 10
        self.sequence_bits = 12

        # Calculate the maximum value
        self.max_worker_id = -1 ^ (-1 << self.worker_id_bits)

        # Shift amounts
        self.worker_id_shift = self.sequence_bits
        self.timestamp_left_shift = self.sequence_bits + self.worker_id_bits
        self.sequence_mask = -1 ^ (-1 << self.sequence_bits)

        self.lock = threading.Lock()

    def _get_timestamp(self):
        return int(time.time() * 1000)

    def generate(self) -> int:
        with self.lock:
            timestamp = self._get_timestamp()

            if timestamp < self.last_timestamp:
                raise Exception("Clock backward exception")

            if timestamp == self.last_timestamp:
                # Within the same millisecond, the sequence number increases
                self.sequence = (self.sequence + 1) & self.sequence_mask
                if self.sequence == 0:
                    # If the sequence number is exhausted, wait for the next millisecond
                    while timestamp <= self.last_timestamp:
                        timestamp = self._get_timestamp()
            else:
                self.sequence = 0

            self.last_timestamp = timestamp

            # Combine the parts and perform bitwise left shift
            new_id = (
                ((timestamp - self.twepoch) << self.timestamp_left_shift)
                | (self.worker_id << self.worker_id_shift)
                | self.sequence
            )

            return new_id


def get_valid_worker_id(max_bits: int = 10) -> int:
    max_worker_id = (1 << max_bits) - 1  # Result of 1023
    worker_id_str = os.getenv("WORKER_ID")

    # Case 1 and Case 2: Environment variable has value
    if worker_id_str:
        try:
            # Case 1: Standard numeric input
            worker_id = int(worker_id_str)
            # Use modulo to force the number to converge within the valid range
            return abs(worker_id) % (max_worker_id + 1)
        except ValueError:
            # Case 2: Input non-numeric string (e.g. "app-worker-a")
            # Use MD5 hash to convert the string to a large integer,
            # then perform modulo convergence
            hash_int = int(hashlib.md5(worker_id_str.encode("utf-8")).hexdigest(), 16)
            return hash_int % (max_worker_id + 1)

    # Case 3: No WORKER_ID is set (forced fallback)
    # Get the hostname of the machine as the唯一性依據
    hostname = socket.gethostname()
    hash_int = int(hashlib.md5(hostname.encode("utf-8")).hexdigest(), 16)

    return hash_int % (max_worker_id + 1)


def generate_id() -> int:
    global _global_generator
    if _global_generator is None:
        _global_generator = Snowflake(worker_id=get_valid_worker_id())
    return _global_generator.generate()
