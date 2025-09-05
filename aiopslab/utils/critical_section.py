import signal
import threading


class CriticalSection:
    def __enter__(self):
        # Only install in the main thread, avoid `ValueError` in worker threads
        self.signaled = False
        if threading.current_thread() is threading.main_thread():
            # Save the original signal handler
            self.original_handler = signal.signal(signal.SIGINT, self.signal_handler)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self, "original_handler"):
            # Restore the original signal handler
            signal.signal(signal.SIGINT, self.original_handler)

            # If SIGINT was raised during the critical section, handle it after the block
            if self.signaled:
                raise KeyboardInterrupt  # Re-raise KeyboardInterrupt to exit

        return False  # Do not suppress exceptions

    def signal_handler(self, signum, frame):
        """Handle SIGINT by just setting a flag to delay it."""
        self.signaled = True  # Flag that SIGINT occurred
        print("\nCtrl+C detected! But deferring the effect for consistency...")
