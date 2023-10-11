import dataclasses

from dockgame.data.parser import ProteinParser


@dataclasses.dataclass
class DecoyGenerator:

    num_decoys: int
    agent_type: str = 'protein'
    parser = ProteinParser()

    def generate_decoys(self, inputs):
        raise ValueError("Subclasses must implement for themselves")
