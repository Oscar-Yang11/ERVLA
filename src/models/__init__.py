from mmengine import Registry

MIMODEL = Registry("MIMODEL")

from src.models.policy_head.DiT import DecoderLayer, TimestepEmbedder
from src.models.projector.linear_projector import MLPProjector
from src.models.runner.base_runner import BaseRunner
from src.models.VLA.P1 import P1
from src.models.VLA.P2 import P2
# from src.models.VLA.P2_L1 import P2_L1
from src.models.VLA.P2_Local import P2_Local
from src.models.VLA.P2_Choice import P2_Choice
from src.models.VLA.P2_Choice_KI import P2_Choice_KI, P2_Local_Choice_KI
from src.models.VLM.qwen3vl import Qwen3VLTextDecoderLayer, Qwen3VLVisionBlock

__all__ = ["BaseRunner", "P1", "P2", "P2_Local", "P2_Choice", "P2_Choice_KI", "P2_Local_Choice_KI"]

AUTO_WRAP_LAYERS = [DecoderLayer, Qwen3VLTextDecoderLayer, Qwen3VLVisionBlock]
