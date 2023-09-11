from .mnist_gp1 import FullyJointLossModel
from .mnist_vexpr_gp import VexprFullyJointLossModel
from .mnist_handson_vexpr_gp import VexprHandsOnLossModel

MODELS = dict(
    FullyJointLossModel=FullyJointLossModel,
    VexprFullyJointLossModel=VexprFullyJointLossModel,
    VexprHandsOnLossModel=VexprHandsOnLossModel,
)
