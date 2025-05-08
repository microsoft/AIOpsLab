import signal


class CriticalSection:
    def __enter__(self):
        # Save the original signal handler
        self.original_handler = signal.signal(signal.SIGINT, self.signal_handler)
        self.signaled = False
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # If SIGINT was raised during the critical section, handle it after the block
        if self.signaled:
            raise KeyboardInterrupt  # Re-raise KeyboardInterrupt to exit

        # Restore the original signal handler
        signal.signal(signal.SIGINT, self.original_handler)
        return False  # Do not suppress exceptions

    def signal_handler(self, signum, frame):
        """Handle SIGINT by just setting a flag to delay it."""
        self.signaled = True  # Flag that SIGINT occurred
        print("\nCtrl+C detected! But deferring the effect for consistency...")
