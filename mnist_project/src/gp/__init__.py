from .botorch_mixed_gp import BotorchMixedGP
from .botorch_partial_handson_gp import BotorchPartialHandsOnGP
from .ol_sequential_joint_gp import OLSequentialJointGP
from .vexpr_handson_gp import VexprHandsOnGP
from .vexpr_joint_gp import VexprFullyJointGP
from .vexpr_partial_handson_gp import VexprPartialHandsOnGP

MODELS = dict(
    BotorchMixedGP=BotorchMixedGP,
    BotorchPartialHandsOnGP=BotorchPartialHandsOnGP,
    OLSequentialJointGP=OLSequentialJointGP,
    VexprFullyJointGP=VexprFullyJointGP,
    VexprHandsOnGP=VexprHandsOnGP,
    VexprPartialHandsOnGP=VexprPartialHandsOnGP,
)
