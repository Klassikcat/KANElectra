import lightning.pytorch as pl


class PretrainingCheckpoint(pl.ModelCheckpoint):
    def __init__(self) -> None:
        super().__init__()  # TODO: Add a checkpointer for generator and discriminator
        
        
class TensorRTCompiler(pl.ModelCheckpoint):
    def __init__(self):
        super().__init__()  # TODO: Create a TensorRT compiler for PyTorch models
    