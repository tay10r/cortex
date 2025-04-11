import json

import cortex


class Server:
    def __init__(self):
        self.api: cortex.API = cortex.API.create_async('wbc_differential',
                                                       max_queue_size=128)
        self.api.setup()

    def close(self):
        self.api.teardown()

    def reset(self, report_id: str, timestamp: str, notes: str) -> dict:
        header = cortex.ReportHeader()
        header.id_ = report_id
        header.timestamp = timestamp
        header.notes = notes
        self.api.reset(header)
        return {}

    def update(self, encoded_image: bytes) -> dict:
        self.api.update(encoded_image)
        return {}

    def finalize(self) -> dict:
        self.api.finalize()
        return {}

    def report(self) -> dict:
        # The results are in JSON
        # Even though it's just going to get converted back to JSON,
        # let's parse it anyway to make sure its correct and pass it
        # back to FastAPI as a dictionary (which is the usual type it gets)
        return json.loads(self.api.results())
