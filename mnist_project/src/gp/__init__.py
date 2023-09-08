from .mnist_gp1 import FullyJointLossModel
from .mnist_vexpr_gp import VexprFullyJointLossModel

MODELS = dict(
    FullyJointLossModel=FullyJointLossModel,
    VexprFullyJointLossModel=VexprFullyJointLossModel,
)
