from .botorch_mixed_gp import BotorchMixedGP
from .botorch_partial_handson_gp import BotorchPartialHandsOnGP
from .mnist_gp1 import FullyJointLossModel
from .mnist_vexpr_gp import VexprFullyJointLossModel
from .mnist_handson_vexpr_gp import VexprHandsOnLossModel
from .vexpr_partial_handson_gp import VexprPartialHandsOnGP

MODELS = dict(
    BotorchMixedGP=BotorchMixedGP,
    BotorchPartialHandsOnGP=BotorchPartialHandsOnGP,
    FullyJointLossModel=FullyJointLossModel,
    VexprFullyJointLossModel=VexprFullyJointLossModel,
    VexprHandsOnLossModel=VexprHandsOnLossModel,
    VexprPartialHandsOnGP=VexprPartialHandsOnGP,
)
