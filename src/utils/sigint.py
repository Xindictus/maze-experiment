from threading import Event


class SigintController:
    """Process-wide shutdown controller."""

    def __init__(self) -> None:
        self._event = Event()

    def request(self) -> None:
        self._event.set()

    def is_requested(self) -> bool:
        return self._event.is_set()


# importable singleton
sigint_controller = SigintController()
